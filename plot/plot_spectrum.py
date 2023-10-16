import torch
import matplotlib.pyplot as plt
import os
import argparse
from colour import Color

parser = argparse.ArgumentParser()
parser.add_argument('-n1', '--name1', help='name of first spectrum')
parser.add_argument('-n2', '--name2', help='name of second spectrum')
parser.add_argument('-o', '--out', help='name of out file')
parser.add_argument('-s', '--skip', help='plot every s values', type=int)
parser.add_argument('-d', '--damping', help='damping', type=float, default=0.0)
parser.add_argument('-i', '--inverse', help='invert spectrum', action='store_true')
parser.add_argument('-l', '--layer', help='layer-wise damping', action='store_true')
args = parser.parse_args()


spec_dir = '/h/nng/projects/OpenOOD/results/ekfac'

specx = torch.load(os.path.join(spec_dir, args.name1 + '_spec.cpt'), map_location='cpu')
specy = torch.load(os.path.join(spec_dir, args.name2 + '_spec.cpt'), map_location='cpu')

if args.layer:
    specx = [sx[::args.skip] + (args.damping * sx.mean()) for sx in specx]
    specy = [sy[::args.skip] + (args.damping * sy.mean()) for sy in specy]
else:
    specx = [sx[::args.skip] + args.damping for sx in specx]
    specy = [sy[::args.skip] + args.damping for sy in specy]

if args.inverse:
    specx = [1. / sx for sx in specx]
    specy = [1. / sy for sy in specy]

sxmin = min([min(sx[torch.nonzero(sx)]).item() if sx.count_nonzero() > 0 else 100 for sx in specx]) / 10
symin = min([min(sy[torch.nonzero(sy)]).item() if sy.count_nonzero() > 0 else 100 for sy in specy]) / 10

print(sxmin)
print(symin)

blue = Color('blue')
colors = list(blue.range_to(Color('green'), len(specx)))

for sx, sy, c in zip(specx, specy, colors):
    plt.scatter(sx, sy, c=str(c), s=2, alpha=0.2)

plt.xlim(left=0)
plt.ylim(bottom=0)
plt.xscale('symlog', linthresh=sxmin)
plt.yscale('symlog', linthresh=symin)

plt.savefig(os.path.join(spec_dir, args.out))

