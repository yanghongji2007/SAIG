# SAIG

##Prepare

```
cd ./pytorch-image-models
pip install -e .
```
Please download CVUSA, CVACT, VIGOR.
Please download the pre-trained model 

|  Dataset   |  Model   | top-1  | weight | log |
|  ---- |  ----  | ----  | ---- | ---- |
|  ImageNet-1k | SAIG-S  | 77.2% | [weight](https://drive.google.com/file/d/1-cXq1ebBRmTLNeHdMtOR53CZkbLqHuCh/view?usp=sharing) |  [log](https://drive.google.com/file/d/1-wDQSbG11n6XOJyDiNaRs4rNVt7gRCEk/view?usp=sharing)|
| ImageNet-1k  | SAIG-D   | 80.3% | [weight](https://drive.google.com/file/d/1kkWxma0lTwK-0PjW4A_3AqpU6Rd7ByWT/view?usp=sharing) | [log](https://drive.google.com/file/d/1Eaz3VT5uH1-3gqEIxDmBaKkAjU-__Qyz/view?usp=sharing) |

## Train
**CVUSA & CVACT**
```shell
bash train.sh
bash train_sam.sh
```
or directly run ```python train.py```  or ```python train_sam.py``` by giving other args
**VIGOR**
```shell
bash train_vigor.sh
bash train_vigor_sam.sh
```

## Test
**CVUSA & CVACT**
```shell
bash test.sh
```
**VIGOR**
```shell
bash test_vigor.sh
```

## Trained model

