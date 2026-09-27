import torch

from nnunetv2.training.loss.skeletonize import Skeletonize
from nnunetv2.training.loss.soft_skeleton import SoftSkeletonize
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.utilities.helpers import softmax_helper_dim1


# =====================================================================
#  Termino clDice puro (Shit et al. 2021). Esta clase SOBREVIVIO al borrado;
#  se conserva tal cual. Devuelve una loss negativa (similaridad a maximizar).
# =====================================================================
class SoftclDiceLoss(torch.nn.Module):
    def __init__(self, iter_=10, smooth=1.):
        super(SoftclDiceLoss, self).__init__()
        self.smooth = smooth

        # Topology-preserving skeletonization:
        # https://github.com/martinmenten/skeletonization-for-gradient-based-optimization
        self.t_skeletonize = Skeletonize(probabilistic=False,
                                         simple_point_detection='EulerCharacteristic')

        # Morphological skeletonization:
        # https://github.com/jocpae/clDice/tree/master/cldice_loss/pytorch
        self.m_skeletonize = SoftSkeletonize(num_iter=iter_)

    def forward(self, y_pred, y_true, t_skeletonize_flage=False):
        """
        Args:
            y_pred (torch.Tensor): Network output (logits) with shape (b, c, x, y(, z)).
            y_true (torch.Tensor): Ground truth labels with shape (b, 1, x, y(, z)).
            t_skeletonize_flage (bool): Enable Topology-preserving skeletonization.
        """
        y_pred_fore = y_pred[:, 1:]
        y_pred_fore = torch.max(y_pred_fore, dim=1, keepdim=True)[0]  # C fg channels -> 1
        y_pred_binary = torch.cat([y_pred[:, :1], y_pred_fore], dim=1)
        y_prob_binary = torch.softmax(y_pred_binary, 1)
        y_pred_prob = y_prob_binary[:, 1]
        with torch.no_grad():
            y_true = torch.where(y_true > 0, 1, 0).squeeze(1).float()
            y_pred_hard = (y_pred_prob > 0.5).float()

            if t_skeletonize_flage:
                skel_pred_hard = self.t_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.t_skeletonize(y_true.unsqueeze(1)).squeeze(1)
            else:
                skel_pred_hard = self.m_skeletonize(y_pred_hard.unsqueeze(1)).squeeze(1)
                skel_true = self.m_skeletonize(y_true.unsqueeze(1)).squeeze(1)
        skel_pred_prob = skel_pred_hard * y_pred_prob
        tprec = (torch.sum(torch.multiply(skel_pred_prob, y_true)) + self.smooth) / \
                (torch.sum(skel_pred_prob) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred_prob)) + self.smooth) / \
                (torch.sum(skel_true) + self.smooth)
        cl_dice_loss = - 2.0 * (tprec * tsens) / (tprec + tsens)
        return cl_dice_loss


# =====================================================================
#  Funciones combinadas (RECONSTRUIDAS tras el borrado).
#
#  Diseno: cada una toma la loss base ESTANDAR de nnU-Net v2 (identica a la del
#  baseline, ver nnUNetTrainer._build_loss) y le SUMA el termino clDice ponderado.
#  Asi la parte base es exactamente la del baseline y la unica diferencia es el
#  termino topologico anadido, manteniendo la comparacion limpia.
#
#  IMPORTANTE: todas asumen enable_deep_supervision=False en el trainer (la red
#  devuelve un unico tensor, no una lista). El termino clDice solo es valido a
#  resolucion completa, que es justo lo que se obtiene con DS desactivado.
# =====================================================================
class dice_cldice_loss(torch.nn.Module):
    """Dice+CE (base nnU-Net) + weight_cldice * clDice."""

    def __init__(self, iter_=3, smooth=1.0, weight_dice=1.0, weight_cldice=0.20,
                 batch_dice=False, ddp=False):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_cldice = weight_cldice
        # base Dice+CE EXACTA de nnU-Net (mismos args que nnUNetTrainer._build_loss)
        self.dc_ce = DC_and_CE_loss(
            {'batch_dice': batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': ddp},
            {}, weight_ce=1, weight_dice=1, ignore_label=None,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        self.cldice = SoftclDiceLoss(iter_=iter_, smooth=smooth)

    def forward(self, net_output, target):
        base = self.dc_ce(net_output, target)
        cl = self.cldice(net_output, target)
        # weight_dice escala la base (1.0 por defecto -> base intacta);
        # clDice (negativo) entra con weight_cldice.
        return self.weight_dice * base + self.weight_cldice * cl


class dice_clCE_loss(torch.nn.Module):
    """Dice+CE (base nnU-Net) + weight_clCE * clDice.

    NOTA: 'clCE' se reconstruye usando el mismo termino clDice (SoftclDiceLoss).
    Si en tu version original 'clCE' era una variante con cross-entropy en los
    terminos de tprec/tsens, avisame y lo ajusto; esta version replica el patron
    de dice_cldice_loss y NO se usa en la fila principal del paper.
    """

    def __init__(self, iter_=3, smooth=1.0, weight_dice=1.0, weight_clCE=1.0,
                 batch_dice=False, ddp=False):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_clCE = weight_clCE
        self.dc_ce = DC_and_CE_loss(
            {'batch_dice': batch_dice, 'smooth': 1e-5, 'do_bg': False, 'ddp': ddp},
            {}, weight_ce=1, weight_dice=1, ignore_label=None,
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        self.cldice = SoftclDiceLoss(iter_=iter_, smooth=smooth)

    def forward(self, net_output, target):
        base = self.dc_ce(net_output, target)
        cl = self.cldice(net_output, target)
        return self.weight_dice * base + self.weight_clCE * cl


class CE_cldice_loss(torch.nn.Module):
    """CE puro (base) + weight_cldice * clDice.

    Variante sin termino Dice en la base: solo cross-entropy + clDice. Recibe un
    dict inicial (compatibilidad con la firma original CE_cldice_loss({}, ...)).
    """

    def __init__(self, ce_kwargs=None, iter_=3, smooth=1.0, weight_ce=1.0,
                 weight_cldice=0.20, ignore_label=None):
        super().__init__()
        if ce_kwargs is None:
            ce_kwargs = {}
        self.weight_ce = weight_ce
        self.weight_cldice = weight_cldice
        self.ignore_label = ignore_label
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.ce = torch.nn.CrossEntropyLoss(**ce_kwargs)
        self.cldice = SoftclDiceLoss(iter_=iter_, smooth=smooth)

    def forward(self, net_output, target):
        target_ce = target[:, 0].long()
        ce = self.ce(net_output, target_ce)
        cl = self.cldice(net_output, target)
        return self.weight_ce * ce + self.weight_cldice * cl


class CE_clCE_loss(torch.nn.Module):
    """CE puro (base) + weight_clCE * clDice. Variante analoga a CE_cldice_loss.

    NOTA: misma observacion que dice_clCE_loss sobre el significado de 'clCE'.
    No se usa en la fila principal del paper.
    """

    def __init__(self, ce_kwargs=None, iter_=3, smooth=1.0, weight_ce=1.0,
                 weight_clCE=1.0, ignore_label=None):
        super().__init__()
        if ce_kwargs is None:
            ce_kwargs = {}
        self.weight_ce = weight_ce
        self.weight_clCE = weight_clCE
        self.ignore_label = ignore_label
        if ignore_label is not None:
            ce_kwargs['ignore_index'] = ignore_label
        self.ce = torch.nn.CrossEntropyLoss(**ce_kwargs)
        self.cldice = SoftclDiceLoss(iter_=iter_, smooth=smooth)

    def forward(self, net_output, target):
        target_ce = target[:, 0].long()
        ce = self.ce(net_output, target_ce)
        cl = self.cldice(net_output, target)
        return self.weight_ce * ce + self.weight_clCE * cl
