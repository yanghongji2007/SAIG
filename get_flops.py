from ptflops import get_model_complexity_info
from ptflops.flops_counter import flops_to_string, params_to_string

from timm.models.SAIG import SAIG_Deep, SAIG_Shallow, resize_pos_embed


def sa_flops(h,w,dim):
    return 2*h*w*h*w*dim

def get_flops(model, input_shape):
    flops, params = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=True)
    _, H, W = input_shape
    SA_flops = sa_flops(H//16, W//16, model.embed_dim)*len(model.blocks)
    flops += SA_flops
    
    return flops_to_string(flops), params_to_string(params)

def main():
    model = SAIG_Deep()
    flops,params = get_flops(model, (3, 224, 224))
    print('flops: ', flops, 'params: ', params)

if __name__ =='__main__':
    main()