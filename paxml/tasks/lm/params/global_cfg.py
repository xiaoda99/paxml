tputype2zone = {'v4': 'us-central2', 'v5': 'us-west4'}

GPT_SPM_PATH = (
    'gs://common_datasets/vocab/c4_en_301_5Mexp_spm.model'  # XD
    # 'gs://mlperf-llm-public2/vocab/c4_en_301_5Mexp2_spm.model'
)
C4_TRAIN_DATADIR = 'gs://common_datasets'  # XD: 'gs://mlperf-llm-public2'
C4_EVAL_DATADIR = 'gs://common_datasets' # XD: 'gs://mlperf-llm-public2'

def strip_zone(gs_path):  # XD
  for zone in tputype2zone.values():
    gs_path = gs_path.replace(f'_{zone}', '')
  return gs_path