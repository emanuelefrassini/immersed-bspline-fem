"""Reproduce the assignment-inspired experiments and save numerical data/plots.

Run from the repository root: python examples/run_experiments.py
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from immersed_bspline_fem import SplineSpace, solve_poisson, uniform_mesh


def save_csv(path, records):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def exp_exact(x):
    return np.expm1(x) / np.expm1(1)


def exp_derivative(x):
    return np.exp(x) / np.expm1(1)


def sine_exact(x):
    return np.sin(2*np.pi*x + 1/3) + x - np.sin(1/3)


def sine_derivative(x):
    return 2*np.pi*np.cos(2*np.pi*x + 1/3) + 1


def sine_source(x):
    return 4*np.pi**2*np.sin(2*np.pi*x + 1/3)


def convergence(out):
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained')
    for p, k in [(1, 0), (2, 1), (3, 2), (3, 0)]:
        for theta, method in [(0, 'strong'), (.5, 'nitsche')]:
            series = []
            for m in (4, 8, 16, 32, 64):
                s = SplineSpace(uniform_mesh(m, theta), p, k)
                sol = solve_poisson(s, lambda x: -exp_derivative(x), (0, 1), method=method)
                l2, h1 = sol.errors(exp_exact, exp_derivative)
                row = dict(degree=p, continuity=k, method=method, theta=theta,
                           elements=m, ndof=s.ndof, h=s.mesh.h, l2=l2, h1=h1)
                if series:
                    prev = series[-1]
                    scale = np.log(prev['h']/row['h'])
                    row.update(l2_rate=np.log(prev['l2']/l2)/scale,
                               h1_rate=np.log(prev['h1']/h1)/scale)
                else:
                    row.update(l2_rate='', h1_rate='')
                series.append(row)
            rows.extend(series)
            color = f'C{[(1,0),(2,1),(3,2),(3,0)].index((p,k))}'
            for ax, field in zip(axes, ('l2', 'h1')):
                ax.loglog([r['h'] for r in series], [r[field] for r in series],
                          'o-' if method == 'strong' else 's--', color=color,
                          label=f'p={p}, C{k}, {method}', markersize=3)
    for ax, label in zip(axes, ('L² error', 'H¹ error')):
        ax.set(xlabel='Background cell size h', ylabel=label)
        ax.grid(True, which='both', alpha=.2)
    axes[0].legend(fontsize=7)
    fig.savefig(out/'convergence.png', dpi=180)
    plt.close(fig)
    save_csv(out/'convergence.csv', rows)


def penalty_study(out):
    s = SplineSpace(uniform_mesh(3, .5))
    x = np.linspace(0, 1, 500)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout='constrained')
    axes[0].plot(x, sine_exact(x), 'k:', linewidth=2, label='Exact')
    for gamma in (.1, 1, 10, 100, 1000, 10000, 100000, 1000000):
        sol = solve_poisson(s, sine_source, (0, 1), method='penalty', gamma=gamma)
        l2, h1 = sol.errors(sine_exact, sine_derivative)
        rows.append(dict(gamma=gamma, condition=sol.condition_number(), l2=l2, h1=h1,
                         boundary_error=np.max(np.abs(sol.evaluate([0, 1]) - [0, 1]))))
        if gamma <= 1000:
            axes[0].plot(x, sol.evaluate(x), label=f'γ={gamma:g}')
    # Plot boundary error against the penalty parameter separately from conditioning.
    ax = axes[1]
    ax.loglog([r['gamma'] for r in rows], [r['boundary_error'] for r in rows],
              'o-', label='Boundary error', color='C0')
    ax.set(xlabel='Absolute penalty γ', ylabel='Max boundary error')
    twin = ax.twinx()
    twin.loglog([r['gamma'] for r in rows], [r['condition'] for r in rows], 's--', color='C1')
    twin.set_ylabel('Condition number', color='C1')
    ax.tick_params(axis='y', labelcolor='C0')
    twin.tick_params(axis='y', labelcolor='C1')
    axes[0].set(xlabel='x', ylabel='Solution', title='Penalty enforcement, 3 cells')
    axes[0].legend(fontsize=8)
    for a in axes:
        a.grid(True, alpha=.2)
    fig.savefig(out/'penalty_study.png', dpi=180)
    plt.close(fig)
    save_csv(out/'penalty_study.csv', rows)


def cut_cell_study(out):
    rows = []
    fig, ax = plt.subplots(figsize=(6.5, 4.5), layout='constrained')
    for method, gamma in [('penalty', 1e6), ('nitsche', 10)]:
        series = []
        for theta in (.001, .01, .1, .5, .9, .99, .999):
            s = SplineSpace(uniform_mesh(6, theta))
            sol = solve_poisson(s, sine_source, (0, 1), method=method, gamma=gamma)
            row = dict(method=method, gamma=gamma, theta=theta,
                       cut_fraction=s.mesh.cut_fractions.min(), condition=sol.condition_number(),
                       scaled_condition=sol.condition_number(scaled=True))
            series.append(row)
        rows.extend(series)
        f = [r['cut_fraction'] for r in series]
        color = 'C0' if method == 'penalty' else 'C1'
        ax.loglog(f, [r['condition'] for r in series], 'o-', color=color, label=method)
        ax.loglog(f, [r['scaled_condition'] for r in series], 's--', color=color,
                  label=f'{method}, diagonal scaling')
    ax.set(xlabel='Smallest physical cell fraction', ylabel='2-norm condition number')
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=.2)
    fig.savefig(out/'cut_conditioning.png', dpi=180)
    plt.close(fig)
    save_csv(out/'cut_conditioning.csv', rows)


def basis_study(out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout='constrained')
    for ax, theta in zip(axes, (0, .5)):
        mesh = uniform_mesh(6, theta)
        s = SplineSpace(mesh)
        x = np.linspace(mesh.nodes[0], mesh.nodes[-1], 700)
        ax.plot(x, s.basis(x).toarray(), linewidth=1.3)
        ax.axvspan(mesh.nodes[0], 0, color='gray', alpha=.15)
        ax.axvspan(1, mesh.nodes[-1], color='gray', alpha=.15)
        for node in mesh.nodes:
            ax.axvline(node, color='gray', linewidth=.6, linestyle=':')
        ax.set(xlabel='x', ylabel='Basis value', title='Fitted' if theta == 0 else 'Immersed (gray = exterior)')
    fig.savefig(out/'basis.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for experiment in (basis_study, convergence, penalty_study, cut_cell_study):
        experiment(args.output)
    print(f'Wrote plots and CSV data to {args.output}')


if __name__ == '__main__':
    main()
