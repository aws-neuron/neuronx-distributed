import neuronxcc.nki.language as nl
from neuronxcc.nki._private.private_api import float4_e2m1fn_x4, float8_e4m3fn_x4

# Hardware config
NUM_QUADRANTS_IN_SBUF = 4
SBUF_QUADRANT_SIZE = 32

# Trn3 quantization block shape for MXFP4/8
_q_height = 8
_q_width = 4

# QMX config
SUPPORTED_QMX_INPUT_DTYPES = [nl.float16, nl.bfloat16]
SUPPORTED_QMX_OUTPUT_DTYPES = [float8_e4m3fn_x4]
MX_SCALE_DTYPE = nl.uint8
SUPPORTED_QMX_OUTPUT_F_DIMS = [128, 512]
SCALE_P_ELEM_PER_QUADRANT = 4
ALLOWED_P_SCALE_IDX_OFFSETS = [0, 4, 8, 12]  # Can fit 4x scales of width 4 in P into partitions 0:16 in each SBUF quadrant

# MMULMX config
# MatmultMX can accept P in [32, 64, 128], corresponding to unpacked P [128, 256, 512]
ALLOWED_P_DIM_MX = [32, 64, 128]
ALLOWED_UNPACKED_P_DIM_MX = [128, 256, 512]

# Gate/up indices
GATE_FUSED_IDX, UP_FUSED_IDX = 0, 1

# Dtype mapping, used to calc correct internal x4 dtype to use for packed weights from framework
N_BTYES_TO_NKI_X4_DTYPE_MAP = {
    2: float4_e2m1fn_x4,
    4: float8_e4m3fn_x4,
}