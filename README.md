# SAIG


## Prepare


```
cd ./pytorch-image-models
pip install -e .
```
Please download CVUSA, CVACT, VIGOR.
Please download the pre-trained model 

|  Dataset   |  Model   | top-1  | weight | log |
|  ---- |  ----  | ----  | ---- | ---- |
|  ImageNet-1k | SAIG-S  | 77.2% | [weight](https://drive.google.com/file/d/1kkWxma0lTwK-0PjW4A_3AqpU6Rd7ByWT/view?usp=sharing) | [log](https://drive.google.com/file/d/1-wDQSbG11n6XOJyDiNaRs4rNVt7gRCEk/view?usp=sharing) |  
| ImageNet-1k  | SAIG-D   | 80.3% | [weight](https://drive.google.com/file/d/1-cXq1ebBRmTLNeHdMtOR53CZkbLqHuCh/view?usp=sharing) |  [log](https://drive.google.com/file/d/1Eaz3VT5uH1-3gqEIxDmBaKkAjU-__Qyz/view?usp=sharing)|

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


CVUSA



|  Method  | Pool   |Loss   | ASAM | R@1 | Weight|
| :------:  |:-----:  | :------:| :------:|:-----:|:-----:|
|  SAIG-S  | GAP    | Triplet| No |88.82 | [Google_Drive](https://drive.google.com/file/d/1ySHh0mVTH0rUpZwGCY6WvcyPd6uwP03D/view?usp=sharing)|
|  SAIG-S  | GAP    | Triplet| Yes | 92.69 |[Google_Drive](https://drive.google.com/file/d/1-od2pxMFGM0uJcnWBBbHxTKSnshKG_lM/view?usp=sharing)  |
|  SAIG-S  | SMD    | Triplet| No | 91.77 | [Google_Drive](https://drive.google.com/file/d/1El6LUsVD9XpSAms0ikhqKw5JCC3unbP-/view?usp=sharing) |
|  SAIG-S  | SMD    | Triplet| Yes | 95.37 | [Google_Drive](https://drive.google.com/file/d/1HFkQgwp-YEFIwZlDbF57FCqHiP0XtnER/view?usp=sharing) |
|  SAIG-D  | GAP    | Triplet| No | 90.29  | [Google_Drive](https://drive.google.com/file/d/1ANQ1HVP7F_y0lDRVn3xg5K_Xs43H5l34/view?usp=sharing) |
|  SAIG-D  | GAP    | Triplet| Yes | 93.97  | [Google_Drive](https://drive.google.com/file/d/188BOFyj9YolOr9XbFi8WKISfsSAmrlcG/view?usp=sharing) |
|  SAIG-D  | SMD    | Triplet| No | 92.71 | [Google_Drive](https://drive.google.com/file/d/1HqfFb69-lsHe6qqwMoN6kKw8RUhdErP4/view?usp=sharing) |
|  SAIG-D  | SMD    | Triplet| Yes | 96.08 |[Google_Drive](https://drive.google.com/file/d/1pA2cHnusLRL1E2A6X85ia-rBAJ5rdAZS/view?usp=sharing)  |

CVACT


|  Method  | Pool   |Loss   | ASAM | R@1 | Weight|
| :------:  |:-----:  | :------:| :------:|:-----:|:-----:|
|  SAIG-S  | GAP    | Triplet| No |81.39 | [Google_Drive](https://drive.google.com/file/d/1SMmN-HFpuDKCkrEfy44ciShduo3cdVmU/view?usp=sharing) |
|  SAIG-S  | GAP    | Triplet| Yes | 85.39 |[Google_Drive](https://drive.google.com/file/d/1BmqG3v2442LTRQedRt5-ghpHIxLZvPBq/view?usp=sharing)  |
|  SAIG-S  | SMD    | Triplet| No | 83.54 | [Google_Drive](https://drive.google.com/file/d/1KMqAunG4z0x8c1y3QsLsh8hqueJNnJcZ/view?usp=sharing) |
|  SAIG-S  | SMD    | Triplet| Yes | 88.44 | [Google_Drive](https://drive.google.com/file/d/1L0-QlbDKxp60lQ1Auo7xupgrpzFhY5S5/view?usp=sharing) |
|  SAIG-D  | GAP    | Triplet| No | 82.40  | [Google_Drive](https://drive.google.com/file/d/1BJD7WaBx2s5tvVxG8nPdNEomBP8ugMZk/view?usp=sharing) |
|  SAIG-D  | GAP    | Triplet| Yes | 86.65  | [Google_Drive](https://drive.google.com/file/d/1RgO8qJu0BiEjUL2eoI9o18asJOd6JbEW/view?usp=sharing) |
|  SAIG-D  | SMD    | Triplet| No | 84.42 | [Google_Drive](https://drive.google.com/file/d/1mHDHmJhJ-UTOWndHy0V8Wp6hAcf0cxS8/view?usp=sharing) |
|  SAIG-D  | SMD    | Triplet| Yes | 89.21 | [Google_Drive](https://drive.google.com/file/d/18-Zf2DsjToLqUbRHJwhzuj-F2PcoSUcq/view?usp=sharing) |


VIGOR


|  Method  | Pool   |Loss   | ASAM | Same Area R@1 | Cross Area R@1 |
| :------:  |:-----:  | :------:| :------:|:-----:|:-----:|
|  SAIG-S  | GAP    | Triplet| No |[40.38](https://drive.google.com/file/d/1B0CBrAxz8Gpbb5zeqcMxjvhgebMABlPM/view?usp=sharing) | [10.22](https://drive.google.com/file/d/17-3bFGc5Dx01jCB9FR4WOVSr3EQcIBtv/view?usp=sharing) |
|  SAIG-S  | GAP    | Triplet| Yes|[46.21](https://drive.google.com/file/d/1Lbhy1MBGMh4w_omNyaUp9ttOHG9jpnPk/view?usp=sharing) | 15.33 |
|  SAIG-D  | GAP | Triplet| No | [42.15](https://drive.google.com/file/d/1yBXADggZzSfZOItsUXBjpYXPwL9ZCQr-/view?usp=sharing)|[11.88](https://drive.google.com/file/d/1kMkG15VyTz8lfWLlGd0-SYUVg2tdSq-3/view?usp=sharing)|
|  SAIG-S  | SMD | Triplet| No | [45.92](https://drive.google.com/file/d/1qzsS0bmc_mDDVVqI15KiHxhkw1UngHOM/view?usp=sharing) | [14.50](https://drive.google.com/file/d/1GDOHSMjaG3_9xEcbOMzhU5gXdFXl5M42/view?usp=sharing) |
|  SAIG-D  | SMD    | Triplet| No | [51.50](https://drive.google.com/file/d/1YKGq0q5NJ5gtqJcoOCmUD1OuZgy5e8js/view?usp=sharing) | [17.58](https://drive.google.com/file/d/1x3MZgFWJzMSOsYzIRux6ZpaZk8rNhOmn/view?usp=sharing) |
|  SAIG-D  | SMD    | InfoNCE| No |  [55.37](https://drive.google.com/file/d/1TXlFq3Ut9wec-CGuov2zpzy5E4_4BqjZ/view?usp=sharing) | [23.47](https://drive.google.com/file/d/1CceYz5x05FU2SdT4N6Ahuu5mIytzh2lm/view?usp=sharing) |
|  SAIG-D  | SMD    | Semi-Triplet| No | [55.60](https://drive.google.com/file/d/1HrlHTPbyNz1MhVkxZ0iT-Ax1AOi5m8h4/view?usp=sharing)  | [22.35](https://drive.google.com/file/d/1IZXDp3CMdwq9A_HLtBHK8vDe6Dh8dqVF/view?usp=sharing) |
|  SAIG-S  | SMD    | Triplet| Yes | [57.57](https://drive.google.com/file/d/1SPXqW09CAP81RU9CaQo-7Rb68-k53XnX/view?usp=sharing)  | [25.44](https://drive.google.com/file/d/17PKE0yo-yYL8zrAlr6KIrQloQbcfku2e/view?usp=sharing)  |
|  SAIG-S  | SMD    | Semi-Triplet| Yes | 62.28 | 30.14 |
|  SAIG-D  | SMD    | Triplet| Yes | [61.27](https://drive.google.com/file/d/1zjRyyWiMAzbxMc9Z5fDEM9yQwrWsHdvA/view?usp=sharing) | [27.61](https://drive.google.com/file/d/1eJ_1vvrPiO-yvA1-xo2Ga6SrGXf5vrTb/view?usp=sharing) |
|  SAIG-D  | SMD    | InfoNCE| Yes | [62.92](https://drive.google.com/file/d/1pBSjQIKIGiHXgAjXz7s7T0doMviSYN_D/view?usp=sharing) | [32.77](https://drive.google.com/file/d/1TomIoZERJrEh_1lzndmmJbNFiOx06mUx/view?usp=sharing) |
|  SAIG-D  | SMD    | Semi-Triplet| Yes | [65.23](https://drive.google.com/file/d/1RmkWDiEqVMZvSAvaQM8RHipRfT_9mq_f/view?usp=sharing) | [33.05](https://drive.google.com/file/d/1phM2Kj3-8L0gwhNSStVJ8JwLmEmB9Uxf/view?usp=sharing) |

