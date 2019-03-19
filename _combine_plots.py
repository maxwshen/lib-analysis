import _config
import sys, os, img2pdf, subprocess
sys.path.append('/home/unix/maxwshen/')
from mylib import util


##
# Functions
##
def pdf_cat(inp_fns, out_fn):
  with open(out_fn, 'wb') as f:
    f.write(img2pdf.convert(inp_fns))
  return

##
# Main
##
@util.time_dec
def main(script = ''):
  print('Combining plots for %s' % (script))
  
  plt_out_dir = _config.OUT_PLACE + script + '/'

  jpgs = [s for s in os.listdir(plt_out_dir) if '.jpg' in s]
  kws = set([jpg_fn.split('_')[-1].replace('.jpg', '') for jpg_fn in jpgs])

  for keyword in kws:
    print(keyword)
    inp_fns = sorted([plt_out_dir + fn for fn in os.listdir(plt_out_dir) if keyword in fn and '.jpg' in fn])
    out_fn = plt_out_dir + '-%s_%s_combined.pdf' % (script, keyword)
    pdf_cat(inp_fns, out_fn)

  return


if __name__ == '__main__':
  if len(sys.argv) > 1:
    main(script = sys.argv[1])
  else:
    print('Provide a script name')