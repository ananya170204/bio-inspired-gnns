"""
Statistical Analysis — Friedman + Nemenyi + Wilcoxon (Holm) + All Figures
=========================================================================
Uses 6 datasets (Twitter excluded due to 59-class extreme imbalance).
4 metrics: accuracy, F1, precision, recall.
Generates:
  - Figure 2: 4-panel CD diagrams
  - Figure 3: Individual vs unified principles (V7 vs V1-V6)
  - Figure 4: Boxplots of macro-F1 across 10 runs
  - Table 6: Average ranks
  - Table 7: Friedman test results
  - Table 8: Wilcoxon with Holm correction

Usage:
    python3 statistical_analysis.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import friedmanchisquare, wilcoxon
from pathlib import Path

try:
    import scikit_posthocs as sp
except ImportError:
    os.system(f"{sys.executable} -m pip install scikit-posthocs --break-system-packages -q")
    import scikit_posthocs as sp

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    os.system(f"{sys.executable} -m pip install statsmodels --break-system-packages -q")
    from statsmodels.stats.multitest import multipletests

# ─────────────────────────────────────────────────────────────────────────────
# Results — 6 datasets (Twitter excluded from statistical analysis)
# 10-seed results
# ─────────────────────────────────────────────────────────────────────────────

DATASETS  = ["Cora", "CiteSeer", "Facebook", "MUTAG", "PROTEINS", "ENZYMES"]
MODELS    = ["C1","C2","C3","C4","C5","C6","C7","C8",
             "V1","V2","V3","V4","V5","V6","V7"]
BASELINES = [m for m in MODELS if m.startswith("C")]
VARIANTS  = [m for m in MODELS if m.startswith("V")]

ACCURACY = {
    "C1": [0.5633, 0.3709, 0.8431, 0.8056, 0.7207, 0.6100],
    "C2": [0.6385, 0.3509, 0.9166, 0.7333, 0.7315, 0.5817],
    "C3": [0.5736, 0.3573, 0.7332, 0.7500, 0.6198, 0.6800],
    "C4": [0.4930, 0.3531, 0.8777, 0.8333, 0.7441, 0.6267],
    "C5": [0.5707, 0.3492, 0.9274, 0.7500, 0.7225, 0.6483],
    "C6": [0.5120, 0.3420, 0.8969, 0.8333, 0.7225, 0.6233],
    "C7": [0.4346, 0.3091, 0.9053, 0.8111, 0.7360, 0.6183],
    "C8": [0.4792, 0.3445, 0.8950, 0.8500, 0.7261, 0.6500],
    "V1": [0.5007, 0.3601, 0.7181, 0.8389, 0.7063, 0.6733],
    "V2": [0.6564, 0.4366, 0.9756, 0.9056, 0.7225, 0.6950],
    "V3": [0.5269, 0.3236, 0.9009, 0.7778, 0.6829, 0.7450],
    "V4": [0.5349, 0.3725, 0.8438, 0.8500, 0.7099, 0.7367],
    "V5": [0.5596, 0.3080, 0.9192, 0.7056, 0.7171, 0.6500],
    "V6": [0.4651, 0.3255, 0.8536, 0.8667, 0.7171, 0.6883],
    "V7": [0.4157, 0.3331, 0.7662, 0.8944, 0.7207, 0.5633],
}

F1 = {
    "C1": [0.5743, 0.3455, 0.6357, 0.7562, 0.7061, 0.5906],
    "C2": [0.6383, 0.3323, 0.8026, 0.6083, 0.7129, 0.5690],
    "C3": [0.5874, 0.3399, 0.4578, 0.6701, 0.5477, 0.6611],
    "C4": [0.5042, 0.3188, 0.6897, 0.8070, 0.7321, 0.6057],
    "C5": [0.5781, 0.3280, 0.7828, 0.6782, 0.7029, 0.6363],
    "C6": [0.5123, 0.3095, 0.7099, 0.7911, 0.7063, 0.6084],
    "C7": [0.4422, 0.2832, 0.7410, 0.7856, 0.7140, 0.6035],
    "C8": [0.4959, 0.3049, 0.7032, 0.8139, 0.7125, 0.6372],
    "V1": [0.5195, 0.3357, 0.5143, 0.8147, 0.6981, 0.6221],
    "V2": [0.6569, 0.4342, 0.9068, 0.8927, 0.7150, 0.6544],
    "V3": [0.5331, 0.2917, 0.7473, 0.7337, 0.6569, 0.7017],
    "V4": [0.5474, 0.3536, 0.6493, 0.8198, 0.7013, 0.6832],
    "V5": [0.5779, 0.2751, 0.8144, 0.5852, 0.7048, 0.6017],
    "V6": [0.4766, 0.2941, 0.6881, 0.8542, 0.7097, 0.6460],
    "V7": [0.4166, 0.2697, 0.5834, 0.8855, 0.7134, 0.5200],
}

PRECISION = {
    "C1": [0.6105, 0.4375, 0.6465, 0.7559, 0.7138, 0.6164],
    "C2": [0.6397, 0.4084, 0.8207, 0.6199, 0.7231, 0.5949],
    "C3": [0.6233, 0.4154, 0.4285, 0.7914, 0.6673, 0.6793],
    "C4": [0.5719, 0.4337, 0.7046, 0.8105, 0.7348, 0.6241],
    "C5": [0.5992, 0.3987, 0.7887, 0.6963, 0.7168, 0.6588],
    "C6": [0.5445, 0.3992, 0.7385, 0.7941, 0.7178, 0.6382],
    "C7": [0.4813, 0.3399, 0.7547, 0.7910, 0.7295, 0.6233],
    "C8": [0.5652, 0.4417, 0.7090, 0.8373, 0.7186, 0.6482],
    "V1": [0.5923, 0.4119, 0.5354, 0.8745, 0.7066, 0.6343],
    "V2": [0.6636, 0.5027, 0.9117, 0.9295, 0.7249, 0.6675],
    "V3": [0.6091, 0.4294, 0.7515, 0.7967, 0.6915, 0.7113],
    "V4": [0.6164, 0.4037, 0.6699, 0.8857, 0.7111, 0.7012],
    "V5": [0.6103, 0.4655, 0.8236, 0.7389, 0.7226, 0.6198],
    "V6": [0.5490, 0.4182, 0.7105, 0.8882, 0.7181, 0.6690],
    "V7": [0.4706, 0.4063, 0.6384, 0.8994, 0.7218, 0.5534],
}

RECALL = {
    "C1": [0.6125, 0.3509, 0.6541, 0.7931, 0.7100, 0.6032],
    "C2": [0.6756, 0.3408, 0.8095, 0.6354, 0.7142, 0.5818],
    "C3": [0.6298, 0.3448, 0.5112, 0.6841, 0.5937, 0.6663],
    "C4": [0.5428, 0.3249, 0.7027, 0.8181, 0.7343, 0.6177],
    "C5": [0.6157, 0.3329, 0.7960, 0.6886, 0.7062, 0.6502],
    "C6": [0.5494, 0.3146, 0.7207, 0.8258, 0.7111, 0.6180],
    "C7": [0.4732, 0.2867, 0.7498, 0.8003, 0.7108, 0.6139],
    "C8": [0.5328, 0.3141, 0.7187, 0.8163, 0.7156, 0.6520],
    "V1": [0.5519, 0.3382, 0.5338, 0.8058, 0.6973, 0.6235],
    "V2": [0.6958, 0.4324, 0.9071, 0.8838, 0.7149, 0.6590],
    "V3": [0.5532, 0.3044, 0.7649, 0.7403, 0.6696, 0.7095],
    "V4": [0.5859, 0.3620, 0.6631, 0.8253, 0.7009, 0.6841],
    "V5": [0.6349, 0.3004, 0.8176, 0.6214, 0.7042, 0.6042],
    "V6": [0.5233, 0.3116, 0.6984, 0.8494, 0.7094, 0.6446],
    "V7": [0.4640, 0.3106, 0.5988, 0.8929, 0.7124, 0.5198],
}

# Per-run data for boxplots (Figure 4) — 10 runs each
RUNS_F1 = {
    "Cora": {
        "C1": [0.5809,0.6103,0.5571,0.5788,0.5663,0.6152,0.5959,0.5233,0.5376,0.5774],
        "C2": [0.5857,0.6328,0.6308,0.7029,0.6432,0.6513,0.6428,0.6687,0.6168,0.6085],
        "C3": [0.5742,0.5951,0.5786,0.6378,0.5513,0.5688,0.5842,0.5977,0.6018,0.5848],
        "C4": [0.5088,0.5323,0.5031,0.5369,0.4570,0.4670,0.4828,0.5564,0.4907,0.5069],
        "C5": [0.5956,0.6067,0.5392,0.5438,0.5384,0.5225,0.6031,0.5843,0.6357,0.6118],
        "C6": [0.5089,0.4983,0.5239,0.5570,0.4703,0.4686,0.5586,0.5372,0.4276,0.5732],
        "C7": [0.4422,0.4491,0.4476,0.4465,0.4576,0.4283,0.4517,0.4426,0.3994,0.4568],
        "C8": [0.5022,0.5212,0.5013,0.4904,0.4450,0.5156,0.5027,0.5120,0.4654,0.5034],
        "V1": [0.5500,0.5119,0.4940,0.5119,0.5145,0.5089,0.5068,0.5133,0.5216,0.5623],
        "V2": [0.6502,0.6769,0.6807,0.6259,0.6288,0.6702,0.6704,0.6620,0.6491,0.6549],
        "V3": [0.5616,0.4853,0.5035,0.6130,0.4934,0.5445,0.5282,0.5371,0.5198,0.5451],
        "V4": [0.5850,0.4266,0.5648,0.5674,0.5782,0.5699,0.5450,0.5433,0.5488,0.5454],
        "V5": [0.6074,0.6328,0.6720,0.4983,0.5486,0.5113,0.6069,0.5836,0.5736,0.5443],
        "V6": [0.5722,0.4653,0.4432,0.4791,0.4967,0.3904,0.5115,0.5233,0.4273,0.4574],
        "V7": [0.5695,0.2727,0.5923,0.2660,0.5539,0.2533,0.5027,0.5469,0.3831,0.2261],
    },
    "CiteSeer": {
        "C1": [0.3036,0.3609,0.3330,0.3949,0.3620,0.3476,0.3409,0.3094,0.3267,0.3760],
        "C2": [0.3354,0.3463,0.3256,0.3143,0.2658,0.3693,0.2999,0.3578,0.3175,0.3910],
        "C3": [0.2998,0.3564,0.3635,0.3073,0.3155,0.3439,0.3576,0.3450,0.3440,0.3663],
        "C4": [0.3093,0.3412,0.2993,0.3109,0.2866,0.3167,0.2522,0.3250,0.3819,0.3649],
        "C5": [0.3141,0.2939,0.3171,0.3866,0.3598,0.3261,0.3017,0.3124,0.3338,0.3350],
        "C6": [0.2866,0.3891,0.2792,0.2856,0.2629,0.2980,0.3353,0.2981,0.3663,0.2935],
        "C7": [0.2915,0.3153,0.2183,0.3169,0.2929,0.2987,0.2823,0.2671,0.2665,0.2823],
        "C8": [0.3481,0.3326,0.3533,0.3119,0.3043,0.2729,0.2727,0.3378,0.2554,0.2606],
        "V1": [0.3418,0.3617,0.3510,0.3151,0.3589,0.3129,0.3398,0.3142,0.3440,0.3181],
        "V2": [0.4640,0.4525,0.3970,0.4586,0.4507,0.4161,0.4170,0.4012,0.4158,0.4694],
        "V3": [0.2925,0.2437,0.3182,0.3235,0.3122,0.3003,0.2901,0.3039,0.2697,0.2629],
        "V4": [0.4000,0.3645,0.3366,0.3274,0.3252,0.3193,0.3885,0.3783,0.3834,0.3132],
        "V5": [0.2108,0.2842,0.2709,0.2391,0.2890,0.2267,0.3251,0.2449,0.3161,0.3442],
        "V6": [0.3397,0.3671,0.2850,0.2314,0.2779,0.2968,0.2932,0.3011,0.2877,0.2609],
        "V7": [0.2642,0.3801,0.2501,0.2359,0.2431,0.2157,0.2959,0.2865,0.3132,0.2121],
    },
    "Facebook": {
        "C1": [0.3455,0.7026,0.5848,0.7133,0.6991,0.6404,0.6689,0.6639,0.6878,0.6504],
        "C2": [0.7531,0.9734,0.6176,0.8991,0.8287,0.8102,0.8121,0.9035,0.6365,0.7915],
        "C3": [0.4566,0.4522,0.4606,0.5249,0.4473,0.4583,0.3809,0.4608,0.4773,0.4595],
        "C4": [0.7213,0.7287,0.7279,0.6978,0.9086,0.4376,0.6932,0.4506,0.8099,0.7213],
        "C5": [0.9042,0.8155,0.7882,0.6124,0.8147,0.7832,0.7903,0.7888,0.7378,0.7932],
        "C6": [0.7659,0.7411,0.6942,0.6865,0.6744,0.6138,0.6996,0.6943,0.8228,0.7063],
        "C7": [0.6469,0.7032,0.7561,0.7795,0.8236,0.7165,0.5854,0.7036,0.8114,0.8840],
        "C8": [0.8672,0.8112,0.6624,0.7230,0.6850,0.7038,0.4366,0.7075,0.6896,0.7457],
        "V1": [0.6951,0.4429,0.3687,0.6914,0.4557,0.7760,0.3032,0.5952,0.6176,0.1974],
        "V2": [0.8276,0.9700,0.9841,0.9015,0.8270,0.8277,0.8288,0.9844,0.9589,0.9580],
        "V3": [0.5305,0.4638,0.8974,0.9800,0.8321,0.8292,0.4641,0.9572,0.7283,0.7902],
        "V4": [0.7771,0.3382,0.6782,0.7864,0.6722,0.6090,0.6748,0.5454,0.6992,0.7122],
        "V5": [0.5427,0.8975,0.8082,0.9524,0.8485,0.8286,0.9679,0.9027,0.7440,0.6512],
        "V6": [0.6876,0.6170,0.7459,0.5495,0.7742,0.7970,0.2517,0.8489,0.7134,0.8957],
        "V7": [0.6082,0.5866,0.6243,0.7139,0.7122,0.1497,0.6818,0.4891,0.5996,0.6690],
    },
    "MUTAG": {
        "C1": [0.8750,0.3793,0.8750,0.8615,0.5179,0.7500,0.9398,0.8831,0.7662,0.7143],
        "C2": [0.7778,0.7500,0.6296,0.4194,0.4000,0.7231,0.6727,0.8286,0.4815,0.4000],
        "C3": [0.7778,0.5179,0.6727,0.6516,0.6786,0.7500,0.5179,0.8831,0.6727,0.5786],
        "C4": [0.9398,0.8194,0.8750,0.6296,0.7231,0.7231,0.8875,0.8831,0.8750,0.7143],
        "C5": [0.9345,0.7143,0.7231,0.6000,0.7778,0.7231,0.3793,0.7662,0.5846,0.5786],
        "C6": [0.8831,0.8036,0.8036,0.8036,0.4000,0.8036,1.0000,0.8194,0.8194,0.7750],
        "C7": [0.9398,0.8194,0.8750,0.5846,0.7231,0.7231,0.8286,0.8194,0.8286,0.7143],
        "C8": [0.9345,0.8750,0.8750,0.6000,0.7231,0.7231,0.8750,0.8831,0.8750,0.7750],
        "V1": [0.8831,0.9398,0.8036,0.7231,0.6990,0.8036,0.8286,0.7231,0.9398,0.8036],
        "V2": [0.9398,0.8750,0.7231,0.8831,0.8036,0.9398,0.9398,0.9398,0.9398,0.9429],
        "V3": [0.8750,0.8875,0.6296,0.7500,0.5846,0.8875,0.4815,0.6990,0.7231,0.8194],
        "V4": [0.9398,0.6296,0.5179,0.8286,0.8286,0.9398,0.8875,0.8036,0.9398,0.8831],
        "V5": [0.7231,0.7231,0.6296,0.7231,0.6296,0.3793,0.6296,0.5179,0.3793,0.5179],
        "V6": [0.8750,0.8036,0.8036,0.8286,0.7662,0.8750,0.8875,0.9398,0.8750,0.8875],
        "V7": [0.9429,0.8831,0.8875,0.9429,0.8875,0.7231,0.9429,0.9429,0.8831,0.8194],
    },
    "PROTEINS": {
        "C1": [0.7153,0.6696,0.7468,0.6873,0.6825,0.6735,0.7093,0.6825,0.7282,0.7656],
        "C2": [0.7595,0.6796,0.7448,0.7197,0.6988,0.7260,0.6168,0.6678,0.7799,0.7364],
        "C3": [0.6080,0.6907,0.5609,0.4973,0.6825,0.5584,0.6664,0.4108,0.3160,0.4861],
        "C4": [0.7571,0.6931,0.7317,0.7281,0.7502,0.7300,0.7332,0.6596,0.7702,0.7679],
        "C5": [0.7045,0.5924,0.7364,0.6997,0.7162,0.7401,0.6823,0.6519,0.7412,0.7637],
        "C6": [0.7153,0.6440,0.7281,0.7582,0.7210,0.6494,0.7132,0.6404,0.7176,0.7758],
        "C7": [0.7486,0.6880,0.6918,0.7146,0.7045,0.7448,0.7148,0.6473,0.7542,0.7316],
        "C8": [0.7127,0.6979,0.7384,0.7400,0.7148,0.6529,0.7113,0.6530,0.7502,0.7532],
        "V1": [0.7332,0.6773,0.7030,0.7048,0.7216,0.6636,0.7113,0.7132,0.6712,0.6821],
        "V2": [0.7236,0.7132,0.7614,0.6963,0.7555,0.7366,0.6570,0.6762,0.6692,0.7614],
        "V3": [0.6741,0.6692,0.7048,0.7259,0.7148,0.5827,0.6937,0.6667,0.4217,0.7153],
        "V4": [0.7431,0.7048,0.6741,0.7467,0.7232,0.6796,0.6880,0.6664,0.6823,0.7045],
        "V5": [0.6664,0.7176,0.6963,0.6881,0.7401,0.6583,0.6963,0.7300,0.7431,0.7113],
        "V6": [0.7232,0.7216,0.6964,0.6988,0.7560,0.7260,0.6664,0.7089,0.7093,0.6907],
        "V7": [0.7548,0.7176,0.6880,0.7281,0.7132,0.7030,0.6712,0.7216,0.7093,0.7270],
    },
    "ENZYMES": {
        "C1": [0.5810,0.5351,0.4898,0.6254,0.5715,0.6597,0.7289,0.5293,0.5490,0.6363],
        "C2": [0.6186,0.5613,0.2833,0.6466,0.5542,0.4600,0.7296,0.5757,0.5509,0.7096],
        "C3": [0.6892,0.7246,0.6418,0.6608,0.7693,0.6154,0.6718,0.5885,0.5971,0.6527],
        "C4": [0.5442,0.4940,0.5069,0.6995,0.6906,0.5690,0.6352,0.6396,0.6201,0.6574],
        "C5": [0.6016,0.6022,0.5834,0.6803,0.6196,0.6964,0.7056,0.6261,0.5873,0.6608],
        "C6": [0.5865,0.5361,0.4476,0.7356,0.5588,0.6053,0.6923,0.6137,0.6343,0.6742],
        "C7": [0.6021,0.5616,0.5777,0.7249,0.4917,0.5424,0.7027,0.6098,0.5637,0.6582],
        "C8": [0.6301,0.5445,0.4887,0.6638,0.7403,0.6214,0.7630,0.6225,0.6572,0.6405],
        "V1": [0.5884,0.6370,0.6609,0.6285,0.4783,0.6165,0.6828,0.6348,0.6312,0.6631],
        "V2": [0.7125,0.5847,0.7018,0.6934,0.6805,0.5919,0.6624,0.6423,0.6217,0.6525],
        "V3": [0.7415,0.7005,0.7296,0.7415,0.6320,0.7276,0.6963,0.7250,0.6645,0.6586],
        "V4": [0.6578,0.6750,0.7263,0.6761,0.6660,0.6720,0.6675,0.6701,0.6766,0.7445],
        "V5": [0.6327,0.4529,0.6813,0.6060,0.5867,0.6255,0.6146,0.6182,0.5840,0.6150],
        "V6": [0.6664,0.6694,0.6321,0.6382,0.5879,0.6201,0.7312,0.6078,0.6410,0.6659],
        "V7": [0.5903,0.4190,0.6430,0.6136,0.4801,0.5126,0.5709,0.3714,0.4976,0.5014],
    },
    # Twitter per-run F1 data (for boxplots only - not used in formal statistical analysis)

}
 

TWITTER_RUNS_F1 = {
    "C1": [0.0361,0.3451,0.2741,0.4248,0.1842,0.3755,0.1697,0.3943,0.1759,0.3048],
    "C2": [0.4351,0.5635,0.5727,0.5818,0.5527,0.5930,0.5853,0.3114,0.5313,0.5912],
    "C3": [0.3182,0.3471,0.0842,0.0577,0.2095,0.0823,0.3529,0.3880,0.2074,0.2070],
    "C4": [0.5024,0.4814,0.3278,0.3825,0.5155,0.5088,0.1183,0.4714,0.4869,0.4917],
    "C5": [0.6102,0.4608,0.6228,0.5100,0.6320,0.5852,0.2784,0.6242,0.4511,0.6370],
    "C6": [0.5229,0.4970,0.3930,0.5004,0.5016,0.4933,0.4724,0.4918,0.4530,0.4148],
    "C7": [0.5147,0.4763,0.3692,0.4907,0.3896,0.4830,0.1920,0.3726,0.4880,0.4628],
    "C8": [0.5237,0.1306,0.4805,0.1686,0.3322,0.5049,0.5186,0.1871,0.3823,0.5019],
    "V1": [0.3979,0.4008,0.2332,0.0284,0.0488,0.0740,0.0354,0.1966,0.0281,0.0872],
    "V2": [0.6187,0.5706,0.6285,0.5794,0.6558,0.6012,0.5698,0.5460,0.5879,0.5796],
    "V3": [0.6141,0.5789,0.6301,0.5154,0.2698,0.6176,0.4139,0.0670,0.0597,0.4143],
    "V4": [0.2837,0.1098,0.1644,0.2712,0.4225,0.1481,0.2534,0.0550,0.3385,0.2716],
    "V5": [0.3474,0.3675,0.4596,0.5194,0.1465,0.2264,0.3643,0.2540,0.3512,0.4716],
    "V6": [0.1991,0.0508,0.2417,0.4305,0.2269,0.1925,0.2706,0.1012,0.4271,0.2497],
    "V7": [0.0821,0.2585,0.0725,0.1116,0.0243,0.0183,0.0514,0.0424,0.2103,0.0318],
}

METRICS = {
    "Accuracy":  ACCURACY,
    "Macro F1":  F1,
    "Precision": PRECISION,
    "Recall":    RECALL,
}

COL_BASE    = "#2196F3"
COL_VARIANT = "#4CAF50"
COL_V2      = "#FF5722"
OUT_DIR = Path("results/statistical")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: build dataframe for one metric
# ─────────────────────────────────────────────────────────────────────────────

def build_df(metric_dict):
    return pd.DataFrame(metric_dict, index=DATASETS).T

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run Friedman + Nemenyi + ranks for one metric
# ─────────────────────────────────────────────────────────────────────────────

def run_stats(metric_name, metric_dict):
    df        = build_df(metric_dict)
    scores    = [df[d].values for d in DATASETS]
    stat, p   = friedmanchisquare(*scores)
    ranks_df  = df.rank(ascending=False, axis=0)
    avg_ranks = ranks_df.mean(axis=1).sort_values()
    nemenyi   = sp.posthoc_nemenyi_friedman(df.values.T)
    nemenyi.index = nemenyi.columns = MODELS
    q_alpha   = 3.391
    CD        = q_alpha * np.sqrt(len(MODELS)*(len(MODELS)+1)/(6*len(DATASETS)))
    sig_pairs = [(MODELS[i], MODELS[j], nemenyi.loc[MODELS[i], MODELS[j]])
                 for i in range(len(MODELS))
                 for j in range(i+1, len(MODELS))
                 if nemenyi.loc[MODELS[i], MODELS[j]] < 0.05]
    print(f"\n{'='*65}")
    print(f" {metric_name}")
    print(f"{'='*65}")
    print(f"  Friedman stat={stat:.4f}  p={p:.2e}  CD={CD:.4f}")
    print(f"  Best: {avg_ranks.idxmin()} ({avg_ranks.min():.2f})  "
          f"Worst: {avg_ranks.idxmax()} ({avg_ranks.max():.2f})")
    print(f"  Nemenyi sig pairs: {len(sig_pairs)}")
    for m1, m2, pv in sig_pairs:
        print(f"    {m1} vs {m2}: p={pv:.4f}")
    return stat, p, CD, avg_ranks, sig_pairs

# ─────────────────────────────────────────────────────────────────────────────
# Wilcoxon — V2 vs each baseline, with Holm correction
# ─────────────────────────────────────────────────────────────────────────────

def run_wilcoxon_all_metrics():
    print(f"\n{'='*65}")
    print(" Wilcoxon — V2 vs Each Baseline (Holm-adjusted, all metrics)")
    print(f" N={len(DATASETS)} datasets, min p = 0.0156")
    print("="*65)
    all_results = {}
    for mname, mdict in METRICS.items():
        v2 = mdict["V2"]
        print(f"\n  [{mname}]")
        raw_pvals = []
        baseline_data = []
        for cfg in BASELINES:
            scores  = mdict[cfg]
            v_wins  = sum(v > b for v, b in zip(v2, scores))
            avg_gap = np.mean([v - b for v, b in zip(v2, scores)])
            try:
                _, p_w = wilcoxon(v2, scores, alternative="greater")
            except Exception:
                p_w = 1.0
            raw_pvals.append(p_w)
            baseline_data.append((cfg, v_wins, avg_gap, p_w))

        # Apply Holm correction
        reject, padj, _, _ = multipletests(raw_pvals, method='holm')

        mresults = []
        for i, (cfg, v_wins, avg_gap, p_w) in enumerate(baseline_data):
            sig = "SIGNIFICANT *" if padj[i] < 0.05 else "not significant"
            mresults.append((cfg, v_wins, avg_gap, p_w, padj[i], sig))
            print(f"    V2 vs {cfg}: {v_wins}/{len(DATASETS)} wins | "
                  f"gap={avg_gap*100:+.1f}% | "
                  f"p_raw={p_w:.4f} | p_adj={padj[i]:.4f} — {sig}")
        all_results[mname] = mresults
    return all_results

# ─────────────────────────────────────────────────────────────────────────────
# Print Table 6: Average Ranks
# ─────────────────────────────────────────────────────────────────────────────

def print_rank_table(all_stats):
    print(f"\n{'='*75}")
    print(" Table 6 — Average Ranks across 6 datasets (lower = better)")
    print(f"{'='*75}")
    print(f"{'Model':<6} {'Accuracy':>10} {'Macro F1':>10} {'Precision':>10} {'Recall':>10}")
    print(f"{'-'*50}")
    for model in MODELS:
        row = []
        for mname in ["Accuracy", "Macro F1", "Precision", "Recall"]:
            _, _, _, avg_ranks, _ = all_stats[mname]
            row.append(f"{avg_ranks[model]:.2f}")
        print(f"{model:<6} {row[0]:>10} {row[1]:>10} {row[2]:>10} {row[3]:>10}")

# ─────────────────────────────────────────────────────────────────────────────
# Print Table 7: Friedman results
# ─────────────────────────────────────────────────────────────────────────────

def print_friedman_table(all_stats):
    print(f"\n{'='*65}")
    print(" Table 7 — Friedman Test Results")
    print(f"{'='*65}")
    print(f"{'Metric':<12} {'Statistic':>12} {'p-value':>14} {'Significant':>12}")
    print(f"{'-'*52}")
    for mname, (stat, p, CD, avg_ranks, _) in all_stats.items():
        sig = "Yes" if p < 0.05 else "No"
        print(f"{mname:<12} {stat:>12.2f} {p:>14.2e} {sig:>12}")

# ─────────────────────────────────────────────────────────────────────────────
# Print Table 8: Wilcoxon with Holm
# ─────────────────────────────────────────────────────────────────────────────

def print_wilcoxon_table(wilcoxon_results):
    print(f"\n{'='*75}")
    print(" Table 8 — Wilcoxon V2 vs Baselines (Holm-adjusted p-values)")
    print(f"{'='*75}")
    print(f"{'Comparison':<12} {'Acc p_adj':>12} {'F1 p_adj':>12} {'Prec p_adj':>12} {'Rec p_adj':>12}")
    print(f"{'-'*60}")
    for i, cfg in enumerate(BASELINES):
        row = []
        for mname in ["Accuracy", "Macro F1", "Precision", "Recall"]:
            padj = wilcoxon_results[mname][i][4]
            marker = "*" if padj < 0.05 else ""
            row.append(f"{padj:.4f}{marker}")
        print(f"V2 vs {cfg:<6} {row[0]:>12} {row[1]:>12} {row[2]:>12} {row[3]:>12}")
    print("  * p < 0.05 (Holm-adjusted)")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: 4-panel CD diagrams
# ─────────────────────────────────────────────────────────────────────────────

def plot_cd_diagrams(all_stats):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()

    for idx, (mname, (stat, p, CD, avg_ranks, sig_pairs)) in enumerate(all_stats.items()):
        ax = axes[idx]
        ax.axis('off')

        ranks_sorted = avg_ranks.sort_values()
        rank_min     = ranks_sorted.min()
        rank_max     = ranks_sorted.max()
        n            = len(MODELS)
        y_axis       = n + 1.5

        ax.set_xlim(rank_min - 1.8, rank_max + 1.8)
        ax.set_ylim(-1, n + 3.5)

        ax.plot([rank_min - 0.5, rank_max + 0.5], [y_axis, y_axis], 'k-', lw=1.5)

        for r in np.arange(int(np.floor(rank_min)), int(np.ceil(rank_max)) + 1):
            ax.plot([r, r], [y_axis - 0.12, y_axis], 'k-', lw=1.2)
            ax.text(r, y_axis - 0.3, str(r), ha='center', va='top', fontsize=8)

        ax.text((rank_min + rank_max)/2, y_axis + 0.5,
                "Average Rank", ha='center', fontsize=9, fontweight='bold')

        ax.annotate('', xy=(ranks_sorted.min() + CD, y_axis + 0.8),
                    xytext=(ranks_sorted.min(), y_axis + 0.8),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
        ax.text(ranks_sorted.min() + CD/2, y_axis + 1.0,
                f"CD={CD:.2f}", ha='center', color='red', fontsize=8, fontweight='bold')

        half  = n // 2
        left  = ranks_sorted.iloc[:half]
        right = ranks_sorted.iloc[half:]

        for i, (model, rank) in enumerate(left.items()):
            y_label = (half - 1 - i) * (n / (half + 1))
            color   = COL_V2 if model == "V2" else (COL_VARIANT if model.startswith("V") else COL_BASE)
            ax.plot([rank, rank], [y_axis, y_label + 0.1], color=color, lw=0.8, ls=':')
            ax.plot(rank, y_axis, 'o', color=color, ms=5, zorder=5)
            ax.text(rank - 0.08, y_label, f"{model} ({rank:.2f})",
                    ha='right', va='center', fontsize=7.5, color=color, fontweight='bold')

        for i, (model, rank) in enumerate(right.items()):
            y_label = (len(right) - 1 - i) * (n / (len(right) + 1))
            color   = COL_V2 if model == "V2" else (COL_VARIANT if model.startswith("V") else COL_BASE)
            ax.plot([rank, rank], [y_axis, y_label + 0.1], color=color, lw=0.8, ls=':')
            ax.plot(rank, y_axis, 'o', color=color, ms=5, zorder=5)
            ax.text(rank + 0.08, y_label, f"{model} ({rank:.2f})",
                    ha='left', va='center', fontsize=7.5, color=color, fontweight='bold')

        drawn = set()
        for i in range(n):
            for j in range(i+1, n):
                m1, m2 = MODELS[i], MODELS[j]
                if abs(avg_ranks[m1] - avg_ranks[m2]) < CD:
                    pair = tuple(sorted([m1, m2]))
                    if pair not in drawn:
                        r1 = min(avg_ranks[m1], avg_ranks[m2])
                        r2 = max(avg_ranks[m1], avg_ranks[m2])
                        ax.plot([r1, r2], [y_axis - 0.07, y_axis - 0.07],
                                'k-', lw=3, alpha=0.2, solid_capstyle='round')
                        drawn.add(pair)

        ax.set_title(f"{mname}\nFriedman stat={stat:.2f}, p={p:.2e} | "
                     f"N={len(DATASETS)} datasets", fontsize=9, pad=8)

    legend_patches = [
        mpatches.Patch(color=COL_V2,      label='V2 — Hierarchical Multi-Scale (best model)'),
        mpatches.Patch(color=COL_VARIANT, label='Other Proposed Variants (V1, V3-V7)'),
        mpatches.Patch(color=COL_BASE,    label='Baseline Configurations (C1-C8)'),
        mpatches.Patch(color='red',       label='Critical Difference (CD, α=0.05)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center',
               bbox_to_anchor=(0.5, 0.01), ncol=2, fontsize=9)

    fig.suptitle(
        "Critical Difference Diagrams (Demšar, 2006)\n"
        f"k={len(MODELS)} models, N={len(DATASETS)} datasets "
        f"(Twitter excluded: 59-class extreme imbalance)",
        fontsize=11, fontweight='bold', y=0.99)

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    plt.savefig(OUT_DIR / "figure2_cd_diagram_4panel.pdf", bbox_inches='tight', dpi=150)
    plt.savefig(OUT_DIR / "figure2_cd_diagram_4panel.png", bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\n  Saved -> {OUT_DIR}/figure2_cd_diagram_4panel.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Individual vs Unified principles (V7 vs V1-V6 and best baseline)
# ─────────────────────────────────────────────────────────────────────────────

def plot_individual_vs_unified(all_stats):
    _, _, _, avg_ranks_acc, _ = all_stats["Accuracy"]
    _, _, _, avg_ranks_f1,  _ = all_stats["Macro F1"]

    variants_of_interest = ["V1","V2","V3","V4","V5","V6","V7"]
    labels = {
        "V1": "V1\nModular",
        "V2": "V2\nHierarchical",
        "V3": "V3\nParallel",
        "V4": "V4\nHub-Aware",
        "V5": "V5\nSmall-World",
        "V6": "V6\nDyn. Rewiring",
        "V7": "V7\nUnified",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, (metric_name, metric_dict, avg_ranks) in zip(axes, [
        ("Accuracy", ACCURACY, avg_ranks_acc),
        ("Macro F1", F1, avg_ranks_f1),
    ]):
        x = np.arange(len(variants_of_interest))
        width = 0.6

        scores = []
        colors = []
        for v in variants_of_interest:
            scores.append(np.mean(metric_dict[v]) * 100)
            if v == "V7":
                colors.append("#FF9800")
            elif v == "V2":
                colors.append(COL_V2)
            else:
                colors.append(COL_VARIANT)

        bars = ax.bar(x, scores, width=width, color=colors,
                      edgecolor='white', linewidth=0.5)

        # Add best baseline line
        best_baseline_score = max(np.mean(ACCURACY[c]) for c in BASELINES) * 100 if metric_name == "Accuracy" else max(np.mean(F1[c]) for c in BASELINES) * 100
        ax.axhline(y=best_baseline_score, color='gray', linestyle='--',
                   linewidth=1.5, label=f'Best baseline ({best_baseline_score:.1f}%)')

        for bar, val in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels([labels[v] for v in variants_of_interest], fontsize=9)
        ax.set_ylabel(f"Mean {metric_name} (%) across 6 datasets", fontsize=10)
        ax.set_title(f"{metric_name}: Individual vs Unified Principles", fontsize=11, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=9)

    legend_patches = [
        mpatches.Patch(color=COL_V2,      label='V2 — Best individual variant'),
        mpatches.Patch(color=COL_VARIANT, label='Other individual variants'),
        mpatches.Patch(color="#FF9800",   label='V7 — Unified (all principles)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center',
               bbox_to_anchor=(0.5, 0.01), ncol=3, fontsize=9)

    fig.suptitle("Figure 3: Individual Biological Principles vs Unified Architecture (V7)\n"
                 "Mean performance across 6 datasets; dashed line = best baseline",
                 fontsize=12, fontweight='bold', y=1.01)

    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    plt.savefig(OUT_DIR / "figure3_individual_vs_unified.pdf", bbox_inches='tight', dpi=150)
    plt.savefig(OUT_DIR / "figure3_individual_vs_unified.png", bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved -> {OUT_DIR}/figure3_individual_vs_unified.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Boxplots of macro-F1 across 10 runs for selected models
# ─────────────────────────────────────────────────────────────────────────────

def plot_boxplots():
    selected = ["C2", "C8", "V2", "V3", "V4", "V7"]
    colors   = {
        "C2": COL_BASE, "C8": COL_BASE,
        "V2": COL_V2,   "V3": COL_VARIANT,
        "V4": COL_VARIANT, "V7": "#FF9800"
    }
    labels = {
        "C2": "C2\nMean+Lin",
        "C8": "C8\nMulti+MLP",
        "V2": "V2\nHierarchical",
        "V3": "V3\nParallel",
        "V4": "V4\nHub-Aware",
        "V7": "V7\nUnified",
    }

    all_datasets = DATASETS + ["Twitter*"]
    all_runs = {**RUNS_F1, "Twitter*": TWITTER_RUNS_F1}

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    for idx, dataset in enumerate(all_datasets):
        ax = axes[idx]
        data = [all_runs[dataset][m] for m in selected]
        bp = ax.boxplot(data, patch_artist=True, notch=False,
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker='o', markersize=4, alpha=0.5))

        for patch, model in zip(bp['boxes'], selected):
            patch.set_facecolor(colors[model])
            patch.set_alpha(0.7)

        ax.set_xticks(range(1, len(selected) + 1))
        ax.set_xticklabels([labels[m] for m in selected], fontsize=8)
        title = dataset
        if dataset == "Twitter*":
            title += "\n(excluded from formal stats)"
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_ylabel("Macro F1", fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide the last unused subplot
    axes[-1].set_visible(False)
    axes[-2].set_visible(False)

    legend_patches = [
        mpatches.Patch(color=COL_BASE,    label='Baseline configurations', alpha=0.7),
        mpatches.Patch(color=COL_V2,      label='V2 — Hierarchical Multi-Scale', alpha=0.7),
        mpatches.Patch(color=COL_VARIANT, label='Other variants (V3, V4)', alpha=0.7),
        mpatches.Patch(color="#FF9800",   label='V7 — Unified', alpha=0.7),
    ]
    fig.legend(handles=legend_patches, loc='lower center',
               bbox_to_anchor=(0.5, 0.01), ncol=4, fontsize=9)

    fig.suptitle("Figure 4: Distribution of Macro-F1 Scores across 10 Independent Runs\n"
                 "Selected baseline and variant architectures across 7 datasets\n"
                 "(*Twitter excluded from formal statistical analysis)",
                 fontsize=12, fontweight='bold')

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    plt.savefig(OUT_DIR / "figure4_boxplots.pdf", bbox_inches='tight', dpi=150)
    plt.savefig(OUT_DIR / "figure4_boxplots.png", bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved -> {OUT_DIR}/figure4_boxplots.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Plot: performance heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_performance_heatmap(avg_ranks_acc):
    df        = build_df(ACCURACY)
    df_sorted = df.loc[avg_ranks_acc.index]
    fig, ax   = plt.subplots(figsize=(10, 11))
    im = ax.imshow(df_sorted.values * 100, cmap='RdYlGn', aspect='auto',
                   vmin=df.values.min()*100, vmax=df.values.max()*100)
    for i in range(len(avg_ranks_acc.index)):
        for j in range(len(DATASETS)):
            val = df_sorted.values[i, j] * 100
            ax.text(j, i, f"{val:.1f}%", ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color="black" if 40 < val < 85 else "white")
    ax.set_xticks(range(len(DATASETS)))
    ax.set_yticks(range(len(avg_ranks_acc.index)))
    ax.set_xticklabels(DATASETS, fontsize=9, rotation=15)
    ax.set_yticklabels(avg_ranks_acc.index, fontsize=9)
    ax.set_title("Test Accuracy (%) — All Models x 6 Datasets\n"
                 "(sorted by average rank)", fontsize=11)
    for ytick, model in zip(ax.get_yticklabels(), avg_ranks_acc.index):
        color = COL_V2 if model == "V2" else (COL_VARIANT if model.startswith("V") else COL_BASE)
        ytick.set_color(color)
        ytick.set_fontweight("bold")
    plt.colorbar(im, ax=ax, label="Test Accuracy (%)", shrink=0.6)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "performance_heatmap.pdf", bbox_inches='tight', dpi=150)
    plt.savefig(OUT_DIR / "performance_heatmap.png", bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved -> {OUT_DIR}/performance_heatmap.pdf")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print(" Statistical Analysis — 6 datasets, 4 metrics, 10 seeds")
print(f" Datasets: {', '.join(DATASETS)}")
print(" (Twitter excluded: 59-class extreme imbalance)")
print("=" * 65)

all_stats = {}
for mname, mdict in METRICS.items():
    stat, p, CD, avg_ranks, sig_pairs = run_stats(mname, mdict)
    all_stats[mname] = (stat, p, CD, avg_ranks, sig_pairs)

wilcoxon_results = run_wilcoxon_all_metrics()

print_rank_table(all_stats)
print_friedman_table(all_stats)
print_wilcoxon_table(wilcoxon_results)

acc_stat, acc_p, _, avg_ranks_acc, _ = all_stats["Accuracy"]

plot_cd_diagrams(all_stats)
plot_individual_vs_unified(all_stats)
plot_boxplots()
plot_performance_heatmap(avg_ranks_acc)

print("\n" + "=" * 65)
print(" Summary across all metrics")
print("=" * 65)
print(f"  {'Metric':<12} {'Stat':>8} {'p-value':>12} {'Best':>6} {'Rank':>6}")
print(f"  {'-'*50}")
for mname, (stat, p, CD, avg_ranks, _) in all_stats.items():
    best = avg_ranks.idxmin()
    print(f"  {mname:<12} {stat:>8.2f} {p:>12.2e} "
          f"{best:>6} {avg_ranks.min():>6.2f}")

print(f"\n  Wilcoxon V2 significant pairs (Holm-adjusted):")
for mname, results in wilcoxon_results.items():
    sig = [r[0] for r in results if r[4] < 0.05]
    print(f"  {mname:<12}: {len(sig)}/8 — {sig}")

print(f"\n  All outputs saved to: {OUT_DIR}/")
print("=" * 65)
