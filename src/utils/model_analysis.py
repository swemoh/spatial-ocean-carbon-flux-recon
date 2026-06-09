import numpy as np
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.ocean_basins import get_region, get_region_ocean_mask_only
from utils.data_loader import load_as_maps, load_for_mlp
from utils.plot_map import plot_map
from utils.color_scheme import feature_colors
from models.unext import CyclicHorizontalPadding2D
from utils.plot_style import set_style_width_10
from matplotlib.patches import Patch


def get_gradient_shap_values(model, X, mask, feature_names=None, max_samples=None, region_id='ARCTIC',month=1):

    # reduce data to region size
    region = get_region(X, region_id)
    
    # add one dimension to mask
    mask_expanded = np.expand_dims(mask, axis=0)

    # reduce mask to region size
    region_mask = np.squeeze(get_region(mask_expanded, region_id))
     
    # only get those samples that belong to the region (T,N,F)
    filtered_region = region[:, region_mask, :]
    filtered_region_merged = filtered_region.reshape(-1, filtered_region.shape[2])
    background = filtered_region_merged[np.random.choice(filtered_region_merged.shape[0], 500, replace=False)]

    selectedData = filtered_region[month]

    rng = np.random.default_rng(42)
    idx = rng.choice(len(selectedData), size=max_samples, replace=False)
    X_eval = selectedData[idx]


    # Create explainer once
    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(X_eval)

    shap_values_array = np.array(shap_values).reshape(max_samples, 15)


    shap_exp = shap.Explanation(values=shap_values_array,
                            data=X_eval,   
                            feature_names=feature_names)


    return shap_exp


def plot_shap_over_time2(model, data, bg_data, mask, feature_names=None, max_samples_per_time=None,
                              region_id='ARCTIC', type='TREND', year=""):

    from utils.plot_style import set_style_width_10
    set_style_width_10() 

    # reduce data to region size
    region = get_region(data, region_id)
    
    # add one dimension to mask
    mask_expanded = np.expand_dims(mask, axis=0)

    # reduce mask to region size
    region_mask = np.squeeze(get_region(mask_expanded, region_id))

    # only get those samples that belong to the region (T,N,F)
    filtered_region = region[:, region_mask, :]

    T, N, F = filtered_region.shape
    num_years = T // 12

    if type == 'TREND':
        filtered_region = filtered_region.reshape(num_years, 12, N, F).reshape(num_years, 12*N, F)
    else:
        filtered_region = filtered_region.reshape(num_years, 12, N, F).transpose(1, 0, 2, 3).reshape(12, num_years*N, F)

    # reduce background to mask only
    bg_data = bg_data[:,mask,:]

    # Create explainer once
    explainer = shap.GradientExplainer(model, bg_data.reshape(-1))

    # Storage for mean |SHAP| per (time, feature)
    mean_abs_shap = np.zeros((T, F), dtype=float)

    rng = np.random.default_rng(42)

    for t in range(T):
        X_t = filtered_region[t]  # (N, F)

        idx = rng.choice(len(X_t), size=max_samples_per_time, replace=False)
        X_t_eval = X_t[idx]


        sv = explainer(X_t_eval)           # sv.values shape: (M, F)
        ab_vals = np.abs(sv.values)           # mean absolute SHAP per feature
        mean_abs_shap[t] = ab_vals.mean(axis=0).reshape(15)
        # mean_shap[t] = sv.value.mean(axis=0)

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    years = np.arange(1958, 1958+num_years)
    
    xlabel = ''
    xseries = []
    
    if type == 'TREND':
        xlabel = "Year"
        xseries = years

    else:
        xlabel = "Month"
        xseries = months
        
    # Plot 1
    plt.figure(figsize=(10, 5))
    for j, name in enumerate(feature_names[:10]):
        plt.plot(xseries, mean_abs_shap[:, j], label=name, color=feature_colors[name])

    plt.xlabel(xlabel)
    plt.ylabel("Mean |SHAP value|")
    plt.legend(
    ncol=3,
    bbox_to_anchor=(0.5, -0.15),
    loc='upper center'
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../../outputs/plots/shap/mlp/' + region_id + '_' + year + '_1.png', format='png', dpi=300,  bbox_inches='tight')
    plt.close()

    # Plot 2
    plt.figure(figsize=(10, 5))
    for j, name in enumerate(feature_names[10:]):
        plt.plot(xseries, mean_abs_shap[:, 10+j], label=name, color=feature_colors[name])

    plt.xlabel(xlabel)
    plt.ylabel("Mean |SHAP value|")
    plt.legend(
    ncol=3,
    bbox_to_anchor=(0.5, -0.15),
    loc='upper center'
    )
    plt.grid(True)   
    plt.tight_layout()
    plt.savefig('../../outputs/plots/shap/mlp/' + region_id + '_' + year + '_2.png', format='png', dpi=300,  bbox_inches='tight')
    plt.close()

    return mean_abs_shap, feature_names

def get_gradient_explainer(model, data, bg_data, mask, feature_names=None, max_samples=None, region_id='ARCTIC'):

    # data has shape (H, W, F)
    # data = np.expand_dims(data, axis=0)

    # reduce data to region size
    # region = get_region(data, region_id)
    # mask_expanded = np.expand_dims(mask, axis=0)
    # region_mask = np.squeeze(get_region(mask_expanded, region_id))
    # filtered_region = region[0, region_mask, :]

    filtered_region = data[mask, :]

    # reduce background to mask only
    bg_data = bg_data[:,mask,:]

    # Create explainer once
    explainer = shap.GradientExplainer(model, bg_data.reshape(-1))

    rng = np.random.default_rng(41)
    idx = rng.choice(len(filtered_region), size=max_samples, replace=False)
    X_eval = filtered_region[idx, :]

    shap_values = explainer.shap_values(X_eval)
    shap_values_array = np.array(shap_values).reshape(max_samples, 15)

    shap_exp = shap.Explanation(values=shap_values_array, data=X_eval, feature_names=feature_names)

    return shap_exp


def heatmap_shap_seasonal(model, data, bg_data, mask, feature_names=None, max_samples_per_time=None,
                              region_id='ARCTIC', year=""):

    from utils.plot_style import set_style_width_10
    set_style_width_10()

    region = get_region(data, region_id)    
    mask_expanded = np.expand_dims(mask, axis=0)
    region_mask = np.squeeze(get_region(mask_expanded, region_id))
    filtered_region = region[:, region_mask, :]

    # TEST
    bg_data =  get_region(data, region_id)
    bg_data =  bg_data[:, region_mask, :]

    T, N, F = filtered_region.shape
    num_years = T // 12

    filtered_region = filtered_region.reshape(num_years, 12, N, F).transpose(1, 0, 2, 3).reshape(12, num_years*N, F)

    # reduce background to mask only
    # bg_data = bg_data[:,mask,:]

    explainer = shap.GradientExplainer(model, bg_data.reshape(-1))

    # Storage for mean |SHAP| per (time, feature)
    mean_abs_shap = np.zeros((T, F), dtype=float)

    rng = np.random.default_rng(42)

    for t in range(T):
        X_t = filtered_region[t]  # (N, F)

        idx = rng.choice(len(X_t), size=max_samples_per_time, replace=False)
        X_t_eval = X_t[idx]

        sv = explainer(X_t_eval)           # sv.values shape: (M, F)
        ab_vals = np.abs(sv.values)           # mean absolute SHAP per feature
        mean_abs_shap[t] = ab_vals.mean(axis=0).reshape(15)

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    xseries = months
        
    plt.figure(figsize=(10, 5))

    # Transpose so that features are on y-axis, months/years on x-axis
    heatmap_data = mean_abs_shap.T  # shape: (F, T)

    means = heatmap_data.mean(axis=1)
    sorted_indices = np.argsort(-means)
    heatmap_data = heatmap_data[sorted_indices]
    feature_names = np.array(feature_names)[sorted_indices]

    im = plt.imshow(
        heatmap_data,
        aspect='auto',
        cmap='Wistia',
        interpolation='nearest',
    )

    plt.colorbar(im, label="Mean |SHAP value|")

    # Axis labels
    plt.xlabel("Month")
    plt.ylabel("Features")

    # Tick labels
    plt.xticks(
        ticks=np.arange(len(xseries)),
        labels=xseries,
        rotation=45,
        ha='right'
    )

    plt.yticks(
        ticks=np.arange(len(feature_names)),
        labels=feature_names
    )

    for i in range(heatmap_data.shape[0]):      # features (rows)
        for j in range(heatmap_data.shape[1]):  # months (columns)
            value = heatmap_data[i, j]
            plt.text(
                j, i, f"{value:.2f}",          
                ha="center", va="center",
                color="black", 
            )

    plt.tight_layout()
    plt.show()
    # plt.savefig(
    #     f'../../outputs/plots/shap/mlp/{region_id}_{year}_heatmap.png',
    #     format='png',
    #     dpi=300,
    #     bbox_inches='tight'
    # )
    plt.close()


# plot mean predicted and true co2 flux over time per basin
def plot_timeseries_analysis(targets, predictions, region, path, start_year):
    region_mask = get_region_ocean_mask_only(region).astype(bool)

    region_targets = get_region(targets, region)
    region_pred = get_region(predictions, region)

    T, H, W = region_targets.shape

    region_targets = region_targets[:,region_mask]
    region_pred = region_pred[:,region_mask]

    yearly_target_mean = region_targets.mean(axis=1)
    yearly_pred_mean = region_pred.mean(axis=1)

    months = np.arange(T)
    years = start_year + months / 12

    plt.figure(figsize=(10, 5))
    plt.plot(years, yearly_target_mean, label='Target', color='blue')
    plt.plot(years, yearly_pred_mean, label='Prediction', color='orange')

    match region:
        case "ARCTIC":
            plt.title("Mean CO2 flux pre in the arctic region")
        case "NORTH_ATLANTIC":
            plt.title("Mean CO2 flux pre in the north atlantic")
        case "EQ_PACIFIC":
            plt.title("Mean CO2 flux pre in the equatorial pacific")
        case "SOUTHERN_OCEAN":
            plt.title("Mean CO2 flux pre in the southern ocean")

    plt.xlabel("Time")
    plt.ylabel("CO2 Flux")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path+'/timeseries_' + region + '.png', format='png', dpi=300,  bbox_inches='tight')


# do the same but just over one year, or summarize by month category
def plot_seasonal_analysis(targets, predictions, region,path):

    region_mask = get_region_ocean_mask_only(region).astype(bool)

    region_targets = get_region(targets, region)
    region_pred = get_region(predictions, region)

    T, H, W = region_targets.shape
    num_years = T // 12

    targets_reshaped = region_targets.reshape(num_years, 12, H, W)
    pred_reshaped = region_pred.reshape(num_years, 12, H, W)

    targets_reshaped = targets_reshaped[:,:,region_mask]
    pred_reshaped = pred_reshaped[:,:,region_mask]

    targ_mean_per_year_month = targets_reshaped.mean(axis=2)
    pred_mean_per_year_month = pred_reshaped.mean(axis=2)

    targ_mean = targ_mean_per_year_month.mean(axis=0)
    pred_mean = pred_mean_per_year_month.mean(axis=0)

    targ_std = targ_mean_per_year_month.std(axis=0, ddof=1)
    pred_std = pred_mean_per_year_month.std(axis=0, ddof=1)

    x = np.arange(12)

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    plt.figure(figsize=(10, 5))
    plt.plot(months, targ_mean, label='Target', color='blue')
    plt.plot(months, pred_mean, label='Prediction', color='orange')

    plt.fill_between(x, targ_mean - targ_std, targ_mean + targ_std, alpha=0.2, color='blue', label='Target ±1σ')
    plt.fill_between(x, pred_mean - pred_std, pred_mean + pred_std, alpha=0.2, color='orange', label='Prediction ±1σ')

    match region:
        case "ARCTIC":
            plt.title("Mean CO2 flux pre in the arctic region")
        case "NORTH_ATLANTIC":
            plt.title("Mean CO2 flux pre in the north atlantic")
        case "EQ_PACIFIC":
            plt.title("Mean CO2 flux pre in the equatorial pacific")
        case "SOUTHERN_OCEAN":
            plt.title("Mean CO2 flux pre in the southern ocean")

    plt.xlabel("Month")
    plt.ylabel("CO2 Flux")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path+'/seasonal_' + region + '.png', format='png', dpi=300,  bbox_inches='tight')

# plot mean and predicted co2 flux averaged over year and plot for multiple years
def plot_yearly_analysis(targets, predictions, region, path, start_year):

    region_mask = get_region_ocean_mask_only(region).astype(bool)

    region_targets = get_region(targets, region)
    region_pred = get_region(predictions, region)

    T, H, W = region_targets.shape
    num_years = T // 12

    targets_yearly = region_targets.reshape(num_years, 12, H, W)
    targets_yearly_flat = targets_yearly[:,:,region_mask]
    yearly_target_mean = targets_yearly_flat.mean(axis=(1, 2))

    pred_yearly = region_pred.reshape(num_years, 12, H, W)
    pred_yearly_flat = pred_yearly[:,:,region_mask]
    yearly_pred_mean = pred_yearly_flat.mean(axis=(1, 2))

    years = np.arange(start_year, start_year+num_years)  
    plt.figure(figsize=(10, 5))
    plt.plot(years, yearly_target_mean, label='Target', color='blue')
    plt.plot(years, yearly_pred_mean, label='Prediction', color='orange')

    match region:
        case "ARCTIC":
            plt.title("Mean CO2 flux pre in the arctic region")
        case "NORTH_ATLANTIC":
            plt.title("Mean CO2 flux pre in the north atlantic")
        case "EQ_PACIFIC":
            plt.title("Mean CO2 flux pre in the equatorial pacific")
        case "SOUTHERN_OCEAN":
            plt.title("Mean CO2 flux pre in the southern ocean")

    plt.xlabel("Year")
    plt.ylabel("CO2 Flux")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path+'/yearly_' + region + '.png', format='png', dpi=300,  bbox_inches='tight')

def complete_model_analysis_map(model_path, dataset_id, start_year=2013, end_year=2018, target_index=3):

    # load model
    model = tf.keras.models.load_model(model_path + "/model.keras")
    
    # load and prepare data
    X_test, Y_test = load_as_maps(start_year=start_year, end_year=end_year, datasets=[dataset_id],target_index=target_index)
    map_mask = X_test[0,:,:, 10] == 1

    scaler = load(model_path + '/scaler.pkl')
    n_samples, h, w, n_features = X_test.shape
    X_test_flat = X_test.reshape(-1,n_features)
    X_test_scaled_flat = scaler.transform(X_test_flat)
    X_test = X_test_scaled_flat.reshape(n_samples, h, w, n_features)

    # run model and evalutate
    predictions = model.predict(X_test)
    predictions = predictions.reshape(n_samples, h, w)
    pred_masked = predictions * map_mask

    pred_flat = predictions.reshape(-1)
    mask = X_test_flat[:, 10] == 1
    pred_flat_masked = pred_flat[mask]

    Y_test_flat = Y_test.reshape(-1)
    Y_test_flat_masked = Y_test_flat[mask]

    general_mse = mean_squared_error(pred_flat_masked, Y_test_flat_masked)
    general_mae = mean_absolute_error(pred_flat_masked, Y_test_flat_masked)
   

    with open(model_path + "/model_evaluation.txt", "a") as f:
        f.write("Evaluation " + dataset_id + ":\n")
        f.write(f"General MSE: {general_mse:.3f}\n")
        f.write(f"General MAE: {general_mae:.3f}\n\n")
        f.write(f"Global RMSE: {np.sqrt(general_mse):.3f}\n\n")

        regions = ['ARCTIC', 'NORTH_ATLANTIC', 'EQ_PACIFIC', 'SOUTHERN_OCEAN']

        for region in regions:
            region_targets = get_region(Y_test, region)
            region_pred = get_region(pred_masked, region)
            region_mask = get_region(X_test[:,:,:, 10] == 1, region)

            region_targets = region_targets[region_mask]
            region_pred = region_pred[region_mask]

            mse = mean_squared_error(region_pred, region_targets)
            mae = mean_absolute_error(region_pred, region_targets)
            f.write(f"MSE {region}: {mse:.3f}\n")
            f.write(f"MAE {region}: {mae:.3f}\n")
            f.write(f"RMSE {region}: {np.sqrt(mse):.3f}\n")
        
        f.write("\n"+  "-" * 30 + "\n")


    # compute error maps
    absolute_error = np.abs(pred_masked - Y_test)
    mean_error = np.mean(absolute_error, axis=0)
    squared_error = (pred_masked - Y_test) ** 2
    rmse = np.sqrt(np.mean(squared_error, axis=0)) 

    # plot error maps
    plot_map(data=mean_error, folder_path=model_path, title='Mean absolute error co2 flux pre reconstruction 2013 - 2018', file_name='mae_' + dataset_id,vmin=0, vmax=1, cmap='summer')
    plot_map(data=rmse, folder_path=model_path, title='Root mean squared error co2 flux pre reconstruction 2013 - 2018', file_name='rmse_' + dataset_id,vmin=0, vmax=1, cmap='summer')

    start_year = 2013

    path = model_path + "/" + dataset_id
    if not os.path.exists(path):
        os.makedirs(path)

    plot_timeseries_analysis(Y_test, pred_masked, "ARCTIC", path, start_year)
    plot_timeseries_analysis(Y_test, pred_masked, "NORTH_ATLANTIC", path, start_year)
    plot_timeseries_analysis(Y_test, pred_masked, "EQ_PACIFIC", path, start_year)
    plot_timeseries_analysis(Y_test, pred_masked, "SOUTHERN_OCEAN", path, start_year)

    plot_seasonal_analysis(Y_test, pred_masked, "ARCTIC", path)
    plot_seasonal_analysis(Y_test, pred_masked, "NORTH_ATLANTIC", path)
    plot_seasonal_analysis(Y_test, pred_masked, "EQ_PACIFIC", path)
    plot_seasonal_analysis(Y_test, pred_masked, "SOUTHERN_OCEAN", path)

    plot_yearly_analysis(Y_test, pred_masked, "ARCTIC", path, start_year)
    plot_yearly_analysis(Y_test, pred_masked, "NORTH_ATLANTIC", path, start_year)
    plot_yearly_analysis(Y_test, pred_masked, "EQ_PACIFIC", path, start_year)
    plot_yearly_analysis(Y_test, pred_masked, "SOUTHERN_OCEAN", path, start_year)

def complete_model_analysis_mlp(model_path, dataset_id, start_year=2013, end_year=2018, target_index=3):

    # load model
    model = tf.keras.models.load_model(model_path + "/model.keras")
    
    # load and prepare data
    X_test_flat, Y_test_flat = load_for_mlp(start_year=start_year, end_year=end_year, datasets=[dataset_id], target_index=target_index)

    mask = X_test_flat[:, 10] == 1
    X_test_flat = np.delete(X_test_flat, [10], axis=1)

    scaler = load(model_path + '/scaler.pkl')
    n_samples, n_features = X_test_flat.shape
    X_test_scaled_flat = scaler.transform(X_test_flat)

    map_samples = int(n_samples/167/360)

    map_mask = mask.reshape(map_samples, 167, 360)
    Y_test = Y_test_flat.reshape(map_samples, 167, 360)

    # run model and evalutate
    pred_flat = model.predict(X_test_scaled_flat)
    pred_flat_masked = pred_flat[mask]

    Y_test_flat_masked = Y_test_flat[mask]

    general_mse = mean_squared_error(pred_flat_masked, Y_test_flat_masked)
    general_mae = mean_absolute_error(pred_flat_masked, Y_test_flat_masked)

    pred_masked = pred_flat.reshape(map_samples, 167, 360) * map_mask
   

    with open(model_path + "/model_evaluation.txt", "a") as f:
        f.write("Evaluation " + dataset_id + ":\n")
        f.write(f"Global MSE: {general_mse:.3f}\n")
        f.write(f"Global MAE: {general_mae:.3f}\n\n")
        f.write(f"Global RMSE: {np.sqrt(general_mse):.3f}\n\n")

        regions = ['ARCTIC', 'NORTH_ATLANTIC', 'EQ_PACIFIC', 'SOUTHERN_OCEAN']

        for region in regions:
            region_targets = get_region(Y_test, region)
            region_pred = get_region(pred_masked, region)
            region_mask = get_region_ocean_mask_only(region).astype(bool)

            region_targets = region_targets[:,region_mask].reshape(-1)
            region_pred = region_pred[:,region_mask].reshape(-1)

            mse = mean_squared_error(region_pred, region_targets)
            mae = mean_absolute_error(region_pred, region_targets)
            f.write(f"MSE {region}: {mse:.3f}\n")
            f.write(f"MAE {region}: {mae:.3f}\n")
            f.write(f"RMSE {region}: {np.sqrt(mse):.3f}\n")
        
        f.write("\n"+  "-" * 30 + "\n")


    # compute error maps
    absolute_error = np.abs(pred_masked - Y_test)
    mean_error = np.mean(absolute_error, axis=0)
    squared_error = (pred_masked - Y_test) ** 2
    rmse = np.sqrt(np.mean(squared_error, axis=0)) 

    # plot error maps
    plot_map(data=mean_error, folder_path=model_path, title='Mean absolute error co2 flux pre reconstruction 2013 - 2018', file_name='mae_' + dataset_id,vmin=0, vmax=1, cmap='summer')
    plot_map(data=rmse, folder_path=model_path, title='Root mean squared error co2 flux pre reconstruction 2013 - 2018', file_name='rmse_' + dataset_id,vmin=0, vmax=1, cmap='summer')

    start_year = 1958

    path = model_path + "/" + dataset_id
    if not os.path.exists(path):
        os.makedirs(path)

    plot_timeseries_analysis(Y_test, pred_masked, "ARCTIC", path, start_year)
    plot_timeseries_analysis(Y_test, pred_masked, "NORTH_ATLANTIC", path, start_year)
    plot_timeseries_analysis(Y_test, pred_masked, "EQ_PACIFIC", path, start_year)
    plot_timeseries_analysis(Y_test, pred_masked, "SOUTHERN_OCEAN", path, start_year)

    plot_seasonal_analysis(Y_test, pred_masked, "ARCTIC", path)
    plot_seasonal_analysis(Y_test, pred_masked, "NORTH_ATLANTIC", path)
    plot_seasonal_analysis(Y_test, pred_masked, "EQ_PACIFIC", path)
    plot_seasonal_analysis(Y_test, pred_masked, "SOUTHERN_OCEAN", path)

    plot_yearly_analysis(Y_test, pred_masked, "ARCTIC", path, start_year)
    plot_yearly_analysis(Y_test, pred_masked, "NORTH_ATLANTIC", path, start_year)
    plot_yearly_analysis(Y_test, pred_masked, "EQ_PACIFIC", path, start_year)
    plot_yearly_analysis(Y_test, pred_masked, "SOUTHERN_OCEAN", path, start_year)

def get_masked_prediction_mlp(model_path, dataset_id, start_year=2013, end_year=2018, target_index=3):

    # load model
    model = tf.keras.models.load_model(model_path + "/model.keras")
    
    # load and prepare data
    X_test_flat, Y_test_flat = load_for_mlp(start_year=start_year, end_year=end_year, datasets=[dataset_id], target_index=target_index)

    mask = X_test_flat[:, 10] == 1
    X_test_flat = np.delete(X_test_flat, [10], axis=1)

    scaler = load(model_path + '/scaler.pkl')
    n_samples, n_features = X_test_flat.shape
    X_test_scaled_flat = scaler.transform(X_test_flat)

    map_samples = int(n_samples/167/360)

    map_mask = mask.reshape(map_samples, 167, 360)
    Y_test = Y_test_flat.reshape(map_samples, 167, 360)

    # run model and evalutate
    pred_flat = model.predict(X_test_scaled_flat,batch_size=8192)

    pred_masked = pred_flat.reshape(map_samples, 167, 360) * map_mask

    return Y_test, pred_masked

def get_masked_prediction_map(model_path, dataset_id, start_year=2013, end_year=2018, target_index=3):

    # load model
    model = tf.keras.models.load_model(model_path + "/model.keras")
    
    # load and prepare data
    X_test, Y_test = load_as_maps(start_year=start_year, end_year=end_year, datasets=[dataset_id],target_index=target_index)
    map_mask = X_test[0,:,:, 10] == 1

    scaler = load(model_path + '/scaler.pkl')
    n_samples, h, w, n_features = X_test.shape
    X_test_flat = X_test.reshape(-1,n_features)
    X_test_scaled_flat = scaler.transform(X_test_flat)
    X_test = X_test_scaled_flat.reshape(n_samples, h, w, n_features)


    # run model and evalutate
    predictions = model.predict(X_test,batch_size=1)
    predictions = predictions.reshape(n_samples, h, w)
    pred_masked = predictions * map_mask

    return Y_test, pred_masked

def plot_yearly_analysis_multi(targets, predictions_dict, path, start_year, dataset_id):
    regions = ["ARCTIC", "NORTH_ATLANTIC", "EQ_PACIFIC", "SOUTHERN_OCEAN"]
    titles = {
        "ARCTIC": "Arctic",
        "NORTH_ATLANTIC": "North Atlantic",
        "EQ_PACIFIC": "Equatorial Pacific",
        "SOUTHERN_OCEAN": "Southern Ocean"
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 6)) 
    axes = axes.flatten()

    T, _, _ = targets.shape
    num_years = T // 12
    years = np.arange(start_year, start_year + num_years)

    for ax, region in zip(axes, regions):
        print("---" + region + "---")
        region_mask = get_region_ocean_mask_only(region).astype(bool)
        # Targets
        region_targets = get_region(targets, region)
        _, H, W = region_targets.shape
        targets_yearly = region_targets.reshape(num_years, 12, H, W)
        targets_yearly_flat = targets_yearly[:,:,region_mask]
        yearly_target_mean = targets_yearly_flat.mean(axis=(1, 2))
        ax.plot(years, yearly_target_mean, label="Target", color="black", linewidth=2)

        # Predictions for each model
        for model_name, preds in predictions_dict.items():
            region_pred = get_region(preds, region)
            pred_yearly = region_pred.reshape(num_years, 12, H, W)
            pred_yearly_flat = pred_yearly[:,:,region_mask]
            yearly_pred_mean = pred_yearly_flat.mean(axis=(1, 2))
            ax.plot(years, yearly_pred_mean, label=model_name, linewidth=1.5)

            sum_diffs = np.sum(np.abs(yearly_target_mean-yearly_pred_mean))
            print(model_name + ": " + f"{sum_diffs:.3f}")

        ax.set_title(titles[region])
        ax.set_xlabel("Year")
        ax.set_ylabel("CO₂ Flux [mol/m²/yr]")
        ax.grid(True)

    # Shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
           loc="lower center",
           bbox_to_anchor=(0.5, 0.01),  
           ncol=len(predictions_dict) + 1)
    # fig.subplots_adjust(hspace=-0.1, wspace=0.2)

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.subplots_adjust(hspace=0.38)
    plt.savefig(path + "/yearly_flux_by_region_" + dataset_id + ".png", format="png", dpi=300, bbox_inches="tight")
    # plt.show()
    plt.close()

def plot_yearly_analysis_multi_presentation(targets, predictions_dict, path, start_year, dataset_id):

    regions = ["ARCTIC", "NORTH_ATLANTIC", "EQ_PACIFIC", "SOUTHERN_OCEAN"]
    titles = {
        "ARCTIC": "Arctic",
        "NORTH_ATLANTIC": "North Atlantic",
        "EQ_PACIFIC": "Equatorial Pacific",
        "SOUTHERN_OCEAN": "Southern Ocean"
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    T, _, _ = targets.shape
    num_years = T // 12
    years = np.arange(start_year, start_year + num_years)

    for ax, region in zip(axes, regions):
        region_mask = get_region_ocean_mask_only(region).astype(bool)
        region_targets = get_region(targets, region)
        _, H, W = region_targets.shape
        targets_yearly = region_targets.reshape(num_years, 12, H, W)
        targets_yearly_flat = targets_yearly[:,:,region_mask]
        yearly_target_mean = targets_yearly_flat.mean(axis=(1, 2))


        ax.plot(years, yearly_target_mean, label="Target", color="black", linewidth=2.5)

        for model_name, preds in predictions_dict.items():
            region_pred = get_region(preds, region)
            pred_yearly = region_pred.reshape(num_years, 12, H, W)
            pred_yearly_flat = pred_yearly[:,:,region_mask]
            yearly_pred_mean = pred_yearly_flat.mean(axis=(1, 2))

            ax.plot(years, yearly_pred_mean, label=model_name, linewidth=2)

        ax.set_title(titles[region], fontsize=16)
        ax.set_xlabel("Year", fontsize=14)
        ax.set_ylabel("CO₂ Flux [mol/m²/yr]", fontsize=14)

        ax.tick_params(axis='both', labelsize=12)

        ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center",
               ncol=len(predictions_dict) + 1,
               fontsize=13,
               title="Model",
               title_fontsize=14)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95]) 
    plt.suptitle("Yearly Mean CO₂ Flux by Region", fontsize=18)

    plt.savefig(path + "/yearly_flux_by_region_" + dataset_id + ".png",
                format="png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_seasonal_analysis_all(targets, model_preds: dict, path, dataset_id):
    set_style_width_10() 

    regions = ["ARCTIC", "NORTH_ATLANTIC", "EQ_PACIFIC", "SOUTHERN_OCEAN"]
    region_titles = {
        "ARCTIC": "Arctic",
        "NORTH_ATLANTIC": "North Atlantic",
        "EQ_PACIFIC": "Equatorial Pacific",
        "SOUTHERN_OCEAN": "Southern Ocean"
    }

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = np.arange(12)

    fig, axes = plt.subplots(len(regions), len(model_preds), figsize=(10, 10))
    axes = np.atleast_2d(axes)

    handles, labels = None, None  # for legend collection

    for r, region in enumerate(regions):
        region_mask = get_region_ocean_mask_only(region).astype(bool)
        # target seasonal cycle for this region
        region_targets = get_region(targets, region)
        T, H, W = region_targets.shape
        num_years = T // 12

        targ_reshaped = region_targets.reshape(num_years, 12, H, W)
        targ_reshaped = targ_reshaped[:,:,region_mask]
        targ_mean_per_year_month = targ_reshaped.mean(axis=2)
        targ_mean = targ_mean_per_year_month.mean(axis=0)
        print(targ_mean)
        targ_std = targ_mean_per_year_month.std(axis=0, ddof=1)

        for c, (model_name, pred) in enumerate(model_preds.items()):
            region_pred = get_region(pred, region)
            pred_reshaped = region_pred.reshape(num_years, 12, H, W)
            pred_reshaped = pred_reshaped[:,:,region_mask]
            pred_mean_per_year_month = pred_reshaped.mean(axis=2)
            pred_mean = pred_mean_per_year_month.mean(axis=0)
            pred_std = pred_mean_per_year_month.std(axis=0, ddof=1)

            ax = axes[r, c]
            line1, = ax.plot(months, targ_mean, label='Target', color='blue')
            line2, = ax.plot(months, pred_mean, label='Prediction', color='orange')

            ax.fill_between(x, targ_mean - targ_std, targ_mean + targ_std, alpha=0.2, color='blue')
            ax.fill_between(x, pred_mean - pred_std, pred_mean + pred_std, alpha=0.2, color='orange')

            ax.set_xticks(x[::3])        
            ax.set_xticklabels(months[::3])

            if c == 0:
                ax.set_ylabel("CO₂ Flux [mol/m²/yr]")
                ax.set_title(region_titles[region], loc="left")
            else:
                ax.set_yticklabels([])

            if r == len(regions)-1:
                ax.set_xlabel("Month")

            ax.grid(True)

            if handles is None and labels is None:
                # Get line handles and labels
                handles, labels = ax.get_legend_handles_labels()

                # Add custom patches for standard deviation
                target_std_patch = Patch(color='blue', alpha=0.2, label='Target ± std')
                pred_std_patch = Patch(color='orange', alpha=0.2, label='Prediction ± std')

                handles.extend([target_std_patch, pred_std_patch])
                labels.extend(['Target ± std', 'Prediction ± std'])

            # column headers = model names
            if r == 0:
                ax.set_title(model_name)

    # global legend below
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02), 
        ncol=len(labels)
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.subplots_adjust(hspace=0.35)
    plt.savefig(f"{path}/seasonal_all_regions_{dataset_id}_.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_seasonal_analysis_all_presentation(targets, model_preds: dict, path, dataset_id):
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure we only use first 3 models
    model_preds = dict(list(model_preds.items())[:3])

    regions = ["ARCTIC", "NORTH_ATLANTIC", "EQ_PACIFIC", "SOUTHERN_OCEAN"]
    region_titles = {
        "ARCTIC": "Arctic",
        "NORTH_ATLANTIC": "North Atlantic",
        "EQ_PACIFIC": "Equatorial Pacific",
        "SOUTHERN_OCEAN": "Southern Ocean"
    }

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x = np.arange(12)

    fig, axes = plt.subplots(len(model_preds), len(regions), figsize=(20, 10))
    axes = np.atleast_2d(axes)

    handles, labels = None, None  # for legend capture

    for r, (model_name, pred) in enumerate(model_preds.items()):
        for c, region in enumerate(regions):

            region_targets = get_region(targets, region)
            T, H, W = region_targets.shape
            num_years = T // 12
            targ_reshaped = region_targets.reshape(num_years, 12, H, W)
            targ_mean_per_year_month = targ_reshaped.mean(axis=(2, 3))
            targ_mean = targ_mean_per_year_month.mean(axis=0)
            targ_std = targ_mean_per_year_month.std(axis=0, ddof=1)

            region_pred = get_region(pred, region)
            pred_reshaped = region_pred.reshape(num_years, 12, H, W)
            pred_mean_per_year_month = pred_reshaped.mean(axis=(2, 3))
            pred_mean = pred_mean_per_year_month.mean(axis=0)
            pred_std = pred_mean_per_year_month.std(axis=0, ddof=1)

            ax = axes[r, c]

            ax.plot(months, targ_mean, label="Target", color="blue", linewidth=2.5)
            ax.plot(months, pred_mean, label=model_name, color="orange", linewidth=2)

            ax.fill_between(x, targ_mean - targ_std, targ_mean + targ_std,
                            alpha=0.2, color="blue")
            ax.fill_between(x, pred_mean - pred_std, pred_mean + pred_std,
                            alpha=0.2, color="orange")

            if r == 0:
                ax.set_title(region_titles[region], fontsize=16)

            if c == 0:
                ax.set_ylabel(model_name + "\n \n CO₂ Flux [mol/m²/yr]", fontsize=16)       
            else:
                ax.set_yticklabels([])

            if r == len(model_preds)-1:
                ax.set_xlabel("Month", fontsize=14)

            ax.tick_params(axis='both', labelsize=12)
            ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.7)

            if handles is None and labels is None:
                handles, labels = ax.get_legend_handles_labels()

    fig.legend(handles, ["Target", "Prediction"],
               loc="lower center",
               ncol=2,
               fontsize=14, title="Legend", title_fontsize=15)

    fig.suptitle("Seasonal CO₂ Flux Across Regions and Models", fontsize=20)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.subplots_adjust(hspace=0.25, wspace=0.15)

    plt.savefig(f"{path}/seasonal_all_regions_{dataset_id}.png", dpi=300, bbox_inches="tight")
    plt.close()
