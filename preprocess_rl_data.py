import json
import sys
import os

sys.path.insert(0, '/root/autodl-tmp/PyVision-RL/verl_agents')
os.chdir('/root/autodl-tmp/PyVision-Image-RL-Data/pyvision_image_rl_data')

from verl.utils.dataset.rl_dataset import transfer_to_rl_form_image_w_mm_hint

data = json.load(open('/root/autodl-tmp/PyVision-Image-RL-Data/pyvision_image_rl_data.json'))
converted = transfer_to_rl_form_image_w_mm_hint(
    data,
    '/root/autodl-tmp/PyVision-RL/verl_agents/verl/utils/dataset/rl_system_prompt_template.json'
)
json.dump(converted, open('/root/autodl-tmp/PyVision-Image-RL-Data/pyvision_image_rl_data_processed.json', 'w'), indent=2)
print(f"Done. {len(converted)} samples")
