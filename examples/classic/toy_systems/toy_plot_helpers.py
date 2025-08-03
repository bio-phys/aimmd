"""
Modfified/extended version of the OpenPathSampling ToyPlot file.
"""

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from openpathsampling.engines.toy import Snapshot


# A little class we use for visualizing these 2D PESs
class CallablePES(object):
    def __init__(self, pes):
        self.pes = pes

    def __call__(self, x, y):
        self.positions = [x, y]
        return self.pes.V(self)

class CallableVolume(object):
    def __init__(self, vol):
        self.vol = vol

    def __call__(self, x, y):
        snapshot = Snapshot(coordinates=np.array([[x,y,0.0]]))
        return 1.0 if self.vol(snapshot) else 0.0

class ToyPlot(object):
    def __init__(self, range_x=None, range_y=None):
        if range_x is None:
            range_x = np.arange(-1.1, 1.1, 0.01)
        if range_y is None:
            range_y = np.arange(-1.1, 1.1, 0.01)
        self.extent = [range_x[0], range_x[-1], range_y[0], range_y[-1]]
        self.X, self.Y = np.meshgrid(range_x, range_y)
        pylab.rcParams['figure.figsize'] = 9, 6
        self.repcolordict = {0 : 'k-', 1 : 'r-', 2 : 'g-', 3 : 'b-',
                             4 : 'y-'}
        self.srcolormap = plt.cm.plasma # the colormap used for the different shooting ranges
        self.spcolormap = plt.cm.RdYlGn # the colormap used for the shooting points
        self.commitor_increments = 50 # the number of increments for commitor values (between 0 and 1)

        self.contour_range = np.arange(0.0, 1.5, 0.1)

        self._states = None
        self._pes = None
        self._interfaces = None
        self._initcond = None
        self._shooting_ranges = None

    def add_shooting_ranges(self, srs):
        if self._shooting_ranges is None:
            self._shooting_ranges = [np.vectorize(CallableVolume(sr))(self.X, self.Y) for sr in srs]
        else:
            self._shooting_ranges.extend([np.vectorize(CallableVolume(sr))(self.X, self.Y) for sr in srs])

    def clear_srs(self):
        self._shooting_ranges = None

    def add_pes(self, pes):
        if self._pes is None:
            self._pes = np.vectorize(CallablePES(pes))(self.X, self.Y)

    def add_states(self, states):
        if self._states is None:
            state = states[0]
            self._states = np.vectorize(CallableVolume(state))(self.X, -self.Y)
            for state in states[1:]:
                self._states += np.vectorize(CallableVolume(state))(self.X, -self.Y)

    def add_interfaces(self, ifaces):
        if self._interfaces is None:
            self._interfaces = []
            for iface in ifaces:
                self._interfaces.append(
                    np.vectorize(CallableVolume(iface))(self.X,self.Y)
                )

    def add_initial_condition(self, initcond):
        self._initcond = initcond

    def plot_pes_initcond(self, trajectories):
        fig, ax = plt.subplots()
        ax.set_xlabel("x", size=14)
        ax.set_ylabel("y", size=14)
        if self._pes is not None:
            plt.contour(self.X, self.Y, self._pes,
                        levels=np.arange(0.0, 1.5, 0.1), colors='k')
        if self._initcond is not None:
            ax.plot(self._initcond.coordinates[0,0],
                    self._initcond.coordinates[0,1],
                    'ro', zorder=3)
        for traj in trajectories:
            plt.plot(traj.coordinates()[:,0,0], traj.coordinates()[:,0,1],
                     self.repcolordict[trajectories.index(traj)],
                     zorder=2)

    def plot(self, trajectories=[], bold=[], rpa=None):
        fig, ax = plt.subplots()
        ax.set_xlabel("x", size=14)
        ax.set_ylabel("y", size=14)
        if self._states is not None:
            plt.imshow(self._states, extent=self.extent, cmap="Blues",
                       interpolation='nearest', vmin=0.0, vmax=2.0,
                       aspect='auto')
        if self._pes is not None:
            plt.contour(self.X, self.Y, self._pes,
                        levels=self.contour_range, colors='k')
        if self._interfaces is not None:
            for iface in self._interfaces:
                plt.contour(self.X, self.Y, iface,
                            colors='r', interpolation='none', levels=[0.5])
        if self._shooting_ranges is not None:
            # proxy artist for legend
            proxy_colors = []
            proxy_names = []
            colors = self.srcolormap(np.linspace(0,1,len(self._shooting_ranges)))
            for i, sr in enumerate(self._shooting_ranges):
                plt.contour(self.X, self.Y, sr,
                            colors=[colors[i]],
                            levels=[0.5])
                proxy_colors.append(plt.Rectangle((0,0),1,1,
                                    fc=colors[i]))
                proxy_names.append('SR {:d}'.format(i))
        if self._initcond is not None:
            ax.plot(self._initcond.coordinates[0,0],
                    self._initcond.coordinates[0,1],
                    'ro', zorder=3)
        for traj in bold:
            ax.plot(traj.xyz[:,0,0], traj.xyz[:,0,1],
                     self.repcolordict[bold.index(traj)], linewidth=2,
                    zorder=1)
        for traj in trajectories:
            ax.plot(traj.xyz[:,0,0], traj.xyz[:,0,1],
                     self.repcolordict[trajectories.index(traj) % 5],
                     zorder=2)
        if rpa is not None:
            color_lookup = np.linspace(0, 1, self.commitor_increments)
            colors = self.spcolormap(color_lookup)
            for sp, count in rpa.items():
                color_idx = np.abs(color_lookup
                                   - float(count['success_path'])/count['total_path']
                                   ).argmin()
                ax.plot(sp.xyz[0,0], sp.xyz[0,1], '^', markersize=8,
                        color=colors[color_idx],
                        zorder=5) # always plot shooting snapshots ontop
            # plot a legend for the commitor colors
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.8])
            cbar_ax.set_title('Committor')
            mpl.colorbar.ColorbarBase(cbar_ax, cmap=self.spcolormap,
                         norm=mpl.colors.Normalize(vmin=0., vmax=1.))
        try:
            plt.legend(proxy_colors, proxy_names, loc=2)
        except NameError:
            pass
        return fig

    def reset(self):
        self._pes = None
        self._interfaces = None
        self._initcond = None
        self._states = None
        self._shooting_ranges = None
