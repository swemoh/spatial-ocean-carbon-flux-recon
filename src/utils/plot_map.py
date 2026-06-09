import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils.plot_style import set_style_width_10

def plot_map(data, folder_path, title, file_name, vmin=0, vmax=1, cmap='viridis'):
    lats = np.linspace(-77, 90, 167)
    lons = np.linspace(-180, 180, 360)
    lon_grid, lat_grid = np.meshgrid(lons, lats)


    fig = plt.figure(figsize=(20, 12))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()

    pcm = ax.pcolormesh(lon_grid, lat_grid, data, transform=ccrs.PlateCarree(), shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    # Set land color to black
    land = cfeature.NaturalEarthFeature('physical', 'land', '110m', edgecolor='face', facecolor='black')
    ax.add_feature(land)

    # Add colorbar
    cbar = plt.colorbar(pcm, ax=ax, orientation='vertical', label='CO₂ Flux (mol/m²/s)')

    plt.title(title)
    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    path = folder_path + '/' + timestamp + '_' + file_name + 'plot.png'
    plt.savefig(path, format='png', dpi=300,  bbox_inches='tight')
    plt.close()

def plot_maps_2x2(data_dict, folder_path, file_name, vmin=0, vmax=1, cmap='YlOrRd'):

    titles = list(data_dict.keys())
    data_list = list(data_dict.values())

    lats = np.linspace(-77, 90, 167)
    lons = np.linspace(-180, 180, 360)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3), subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    pcm_list = []
    for i, ax in enumerate(axes):
        pcm = ax.pcolormesh(lon_grid, lat_grid, data_list[i],
                            transform=ccrs.PlateCarree(), shading='auto',
                            cmap=cmap, vmin=vmin, vmax=vmax)

        land = cfeature.NaturalEarthFeature('physical', 'land', '110m',
                                            edgecolor='face', facecolor='black')
        ax.add_feature(land)
        ax.set_global()
        ax.set_title(titles[i])
        ax.tick_params(axis='both', which='major')

        pcm_list.append(pcm)

    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(pcm_list[-1], cax=cbar_ax)
    cbar.set_label(
    'RMSE of CO₂ Flux [mol/m²/year]',
    labelpad=15 
    )

    plt.tight_layout(rect=[0, 0, 0.9, 1])  

    path = f"{folder_path}/{file_name}.png"
    plt.savefig(path, format='png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_maps_4x3_target_reconst_error(
    data_dict, folder_path, file_name,
    vmin1=0, vmax1=1, 
    vmin2=0, vmax2=1
):
    """
    Plots 12 global maps (4x3 grid):
    - Columns 1–2 share one colorbar (cmap1, vmin1/vmax1)
    - Column 3 shares a separate colorbar (cmap2, vmin2/vmax2)
    """

    set_style_width_10()

    cmap1='bwr'
    cmap2='bwr'

    titles = list(data_dict.keys())
    data_list = list(data_dict.values())

    if len(data_list) != 12:
        raise ValueError("data_dict must contain exactly 12 maps.")

    lats = np.linspace(-77, 90, 167)
    lons = np.linspace(-180, 180, 360)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    fig, axes = plt.subplots(4, 3, figsize=(10, 8),
                             subplot_kw={'projection': ccrs.PlateCarree()})
    axes = axes.flatten()

    pcm_list_col12 = [] 
    pcm_list_col3 = []   

    for i, ax in enumerate(axes):
        col = i % 3 

        if col < 2:
            pcm = ax.pcolormesh(
                lon_grid, lat_grid, data_list[i],
                transform=ccrs.PlateCarree(), shading='auto',
                cmap=cmap1, vmin=vmin1, vmax=vmax1
            )
            pcm_list_col12.append(pcm)
        else:
            pcm = ax.pcolormesh(
                lon_grid, lat_grid, data_list[i],
                transform=ccrs.PlateCarree(), shading='auto',
                cmap=cmap2, vmin=vmin2, vmax=vmax2
            )
            pcm_list_col3.append(pcm)

        ax.add_feature(
            cfeature.NaturalEarthFeature('physical', 'land', '110m',
                                         edgecolor='face', facecolor='black')
        )
        ax.set_global()

        ax.set_title(titles[i])
        ax.tick_params(axis='both')

    cbar_ax1 = fig.add_axes([0.1, 0.08, 0.35, 0.02])  # [left, bottom, width, height]
    cbar1 = fig.colorbar(pcm_list_col12[0], cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label(' CO₂ Flux [mol/m²/year]')

    # Colorbar for third column
    cbar_ax2 = fig.add_axes([0.55, 0.08, 0.35, 0.02])
    cbar2 = fig.colorbar(pcm_list_col3[0], cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('Prediction Error [mol/m²/year]')


    plt.subplots_adjust(
        wspace=-0.12,  
        hspace=0.2,
        left=0.03, right=0.97, top=0.95, bottom=0.12
    )

    path = f"{folder_path}/{file_name}.png"
    plt.savefig(path, format='png', dpi=400, bbox_inches='tight')
    plt.show()
    plt.close()