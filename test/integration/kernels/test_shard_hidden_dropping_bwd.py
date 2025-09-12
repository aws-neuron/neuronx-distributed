import logging
from test_utils import generate_kernel_parameters
import json
from test_shard_hidden_dropping import moe_blockwise_bwd
import sys
import copy

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

if __name__ == "__main__":
    """
    Runs the Backward Integration test for the Shard Hidden Dropping Kernel
    Args:
      sys.argv[1]: Test Grid File Name
      sys.argv[2]: Index of the test case to run
      sys.argv[3]: Outut file name for the results

    """
    if len(sys.argv) < 2:
      raise Exception("Test grid file not provided")
    
    file = sys.argv[1]
    with open(file, 'r') as f:
      test_grid = json.load(f)
    index = int(sys.argv[2])

    logger.info(f"Running Test Case from {file}, {test_grid[index]}")
    
    identifier = 'NKI_BWD_INTEGRATION_TEST'
    metadata = ""
    if len(metadata) != 0:
      name, metadata = metadata

    saved_results = []
    PAYLOAD_TEMPLATE = {
        "cfg": {
            "TestDef": {
                'TestName': identifier, 'Identifier': identifier, 'ModelName': 'MOE_SHARD_HIDDEN_DROPPING', 'InferenceOrTraining': 'Training',
            },
            "TestOpt": {"CompilerFlags":"No Flags Used"},
            "Hardware": {},
            "Software": {}
        },
        'kpi': {},
        'misc': {'ModelCategory': 'Not categorized'},
        "artifact": {},
    }
    test_line = test_grid[index]
    params = generate_kernel_parameters(test_line)
    logger.info(f"Running Line, {test_line}")
    if len(params) == 0:
        logger.warning(f"{test_line} skipped as parameters could not be generated")
    
    for param in params:
      payload = copy.deepcopy(PAYLOAD_TEMPLATE)
      payload['kpi']['TestPassFail'] = 'Pass'
      header, model, sweep_name, H, T, E, TOPK, I_TP, BS, dtype, dma_skip, rtype, cf, EP, fail = param
      header = str(header)
      log_header = f"model={model}, sweep_name={sweep_name}, Hidden={H}, T={T}, Experts={E}, TopK={TOPK}, I_TP={I_TP}, BatchSize={BS}, dtype={dtype}, skip={dma_skip}, router={rtype}, capacityfactor={cf}, EP={EP}, xfail={fail}"
      fail_log_header = f"{model},{sweep_name},{H},{T},{E},{TOPK},{I_TP},{BS},{dtype},{dma_skip},{rtype},{cf},{EP},{fail}"

      payload['cfg']['TestDef']['Custom'] = dict()
      payload['cfg']['TestDef']['Custom']['Hidden'] = test_line['H']
      payload['cfg']['TestDef']['Custom']['T'] = test_line['T']
      payload['cfg']['TestDef']['Custom']['Experts'] = test_line['E']
      payload['cfg']['TestDef']['Custom']['TopK'] = test_line['TOPK']
      payload['cfg']['TestDef']['Custom']['Itp'] = test_line['Intermediate']
      payload['cfg']['TestDef']['Custom']['Bs'] = test_line['BS']
      payload['cfg']['TestDef']['Custom']["TpDegree"] = test_line['TP_degree']
      payload['cfg']['TestDef']['Custom']['Dtype'] = str(dtype)
      payload['cfg']['TestDef']['Custom']['Skip'] = dma_skip
      payload['cfg']['TestDef']['Custom']['Rtype'] = rtype
      payload['cfg']['TestDef']['Custom']['Cf'] = cf
      payload['cfg']['TestDef']['Custom']['Ep'] = test_line['EP_degree']
      payload['cfg']['TestDef']['Custom']['ModelType'] = model
      payload['cfg']['TestDef']['Custom']['SweepName'] = sweep_name

      logger.info(f"Running {log_header}")
      if fail:
        payload["kpi"]["TestPassFail"] = "Fail"
        payload['kpi']["TestFailMessage"] = "xfail: Unable to allocate memory for goldens"
        logger.error(f"{fail_log_header},xfail Unable to allocate memory for goldens")
      else:
        try: 
          kpi_data = moe_blockwise_bwd(H, T, E, TOPK, I_TP, BS, dtype, dma_skip, rtype, cf, EP, 0)
          payload['cfg']["TestDef"]["Custom"] = {**payload['cfg']["TestDef"]["Custom"], **kpi_data}
          payload['kpi']["TestFailMessage"] = "Pass"
          logger.info(f"{test_line}, {log_header}, {str(payload['cfg']['TestDef']['Custom'])}\n")
        except Exception as err:
          payload["kpi"]["TestPassFail"] = "Fail"
          payload['kpi']["TestFailMessage"] = str(err)
          logger.error(f"{test_line}, {log_header}, {str(err)}\n")
      
      saved_results.append(payload)

    logging.info("All Test Cases Passed")

    if len(sys.argv) == 4:
      with open(sys.argv[3], "w") as f:
        json.dump(saved_results, f)
