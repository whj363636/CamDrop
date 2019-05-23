import os
import json
import time
import sys


def find_checkpoint(path):
  try:
    with open(os.path.join(path, 'stats.json')) as f:
      js = json.load(f)
      start = js[0]['epoch_num']
      global_step = js[0]['global_step']
  except IOError:
    start, global_step = None, None
  return start, global_step

def find_gpus(need):
  os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
  memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]

  free = 0
  gpu_list = ''
  for i, mem in enumerate(memory_gpu):
    if mem > 12000:
      free += 1
      gpu_list = gpu_list + str(i) + ','
    if free == need:
      break
  gpu_list = gpu_list[:-1]
  print(memory_gpu)
  os.system('rm tmp')
  return gpu_list, free

if __name__ == '__main__':
  keep_prob = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
  # lr = [i/100.0 for i in range(1500, 3, -20)]
  lr = [15, 13, 10, 7, 5, 3, 1, 0.7, 0.5, 0.3, 0.1, 0.07, 0.05, 0.03]
  flag = True

  while flag:
    need = int(sys.argv[1])
    gpu_list, free = find_gpus(need)

    if free == need:
      cmd = sys.argv[2:]

      for i in keep_prob:
        for j in lr:
          tmp_cmd = ''.join(cmd) + ' --keep_prob %.2f' % i
          tmp_cmd = ''.join(tmp_cmd) + ' --lr %.2f' % j
          os.system(tmp_cmd)
      flag = False

    else:
      time.sleep(20)

