"""
Interface to the HDF5 database
"""
import datetime

import numpy as np
import pandas
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns


class LCEImage(object):
    
    def __init__(self, dbase_path, top_level='/'):
        self.dbase_path = dbase_path
        self.top_level = top_level
        self.level_flag = len(list(filter(None, self.top_level.split('/'))))
        
    @property
    def meta(self):
        with h5py.File(self.dbase_path, 'r') as hf:
            grp = hf[self.top_level]
            attrs = dict(grp.attrs)
        
        return attrs
    
    def peek(self, save2file=False):
        if 'image' not in self:
            return ValueError('Can only plot image datasets.')
        fig = plt.figure()
        ax = fig.gca()
        ax.imshow(self['image'])
        ax.set_xticks([])
        ax.set_yticks([])
        # plot curves
        for i, (x, y, r) in enumerate(zip(self['x_center'],
                                          self['y_center'],
                                          self['physical_radii'])):
            circ = Circle((x, y), r/self.meta['physical_conversion'],
                          facecolor='none',
                          edgecolor=sns.color_palette('deep')[i],
                          fill=False,
                          lw=2,
                          label=r'$R={:.2f}$ mm'.format(r))
            ax.add_patch(circ)
        # plot scale
        circ = Circle((self.meta['x_center_scale'], self.meta['y_center_scale']),
                      self.meta['physical_radius_scale']/self.meta['physical_conversion'],
                      facecolor='none', edgecolor='k', fill=False,lw=2,
                      label=r'$R={:.2f}$ mm'.format(self.meta['physical_radius_scale']))
        ax.add_patch(circ)
        ax.legend(loc='best', frameon=True, ncol=2)
        if save2file:
            fig.savefig(save2file, bbox_inches='tight')
    
    def to_dataframe(self):
        if 'image' not in self:
            return pandas.concat([k.to_dataframe() for k in self]).reset_index()
        df_dict = self.meta
        df_dict['date'] = datetime.datetime.strptime(list(filter(None,self.top_level.split('/')))[0],'%m_%d_%y')
        df_dict['physical_radii'] = self['physical_radii']
        df_dict['strain'] = self['strain']
        if 'cylinder_radius' in self.meta:
            df_dict['fixity'] = self['physical_radii']/self.meta['cylinder_radius']
        
        return pandas.DataFrame(df_dict)
    
    def __repr__(self):
        with h5py.File(self.dbase_path, 'r') as hf:
            dset_names = [k for k in hf[self.top_level].keys()]
        dset_names = '\n'.join(dset_names)
        attr_names = '\n'.join(['{}'.format(k) for k in self.meta])
        return '''{}
Datasets
--------
{}

Attributes
----------
{}
'''.format(self.top_level, dset_names, attr_names)
        
    def __getitem__(self, key):
        if self.level_flag == 0:
            if type(key) is int:
                with h5py.File(self.dbase_path, 'r') as hf:
                    dates = [k for k in hf.keys()]
                dates = sorted(dates, key=lambda x: datetime.datetime.strptime(x, '%m_%d_%y'))
                key = dates[key]
        elif self.level_flag == 1:
            key = str(key)
        with h5py.File(self.dbase_path, 'r') as hf:
            grp = hf[self.top_level]
            if key not in grp:
                raise IndexError('{} not found in {}'.format(key, self.top_level))
            ds = grp[key]
            if isinstance(ds, h5py.Group):
                return LCEImage(self.dbase_path, top_level='/'.join([self.top_level, key]))
            else:
                return np.array(ds)
            
    def __contains__(self, key):
        with h5py.File(self.dbase_path, 'r') as hf:
            flag = key in hf[self.top_level]
        return flag
            