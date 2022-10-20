from ptflops import get_model_complexity_info
from timm.models.swin_transformer_GCA import swin_tiny_patch4_window7_224, swin_tiny_patch4_window7_224_GCA
from timm.models.PVT import pvt_small
from timm.models.vision_transformer_GCA import vit_base_patch16_224,vit_small_patch16_224,vit_tiny_patch16_224
#from timm.models.vision_transformer_mlp_stem import vit_small_patch16_224
from timm.models.vision_transformer_hybrid import vit_base_r50_s16_224, vit_small_resnet50d_s16_224
from timm.models.ResT import rest_small,rest_lite
import torchvision
#model = torchvision.models.vision_transformer(pretrained=False)

model = vit_small_patch16_224()
flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
print('flops: ', flops, 'params: ', params)
