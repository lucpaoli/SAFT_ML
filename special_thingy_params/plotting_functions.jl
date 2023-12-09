function average_percentage_deviation(predicted_prop, exptl_prop)

    # Check if the vectors have the same length
    length(predicted_prop) == length(exptl_prop) || throw(ArgumentError("Vectors must have the same length"))

    # Calculate the absolute percentage deviation for each pair of corresponding elements
    absolute_percentage_deviations = abs.((predicted_prop .- exptl_prop) ./ exptl_prop) * 100

    # Calculate the average percentage deviation
    average_percentage_deviation = mean(absolute_percentage_deviations)

    return average_percentage_deviation
end;

function average_absolute_deviation(predicted_prop, exptl_prop)

    # Check if the vectors have the same length
    length(predicted_prop) == length(exptl_prop) || throw(ArgumentError("Vectors must have the same length"))

    # Calculate the absolute percentage deviation for each pair of corresponding elements
    absolute_deviations = abs.(predicted_prop .- exptl_prop)

    # Calculate the average percentage deviation
    average_absolute_deviation = mean(absolute_deviations)

    return average_absolute_deviation
end;

function process_raw_params(file_path, pcp_source_params, last_n_rows=0, start_row_interval=0, end_row_interval=0)
    
    raw_params = CSV.read(file_path, DataFrame, header=1)

    # create processed_data dataframe
    rename!(raw_params, :Column9 => :split)
    rename!(raw_params, :name => :species)
    unique_values_raw_params = unique(raw_params[!, :species])
    num_rows_raw_params = length(unique_values_raw_params)

    processed_data = DataFrame(
        species = fill(missing, num_rows_raw_params),
        isomeric_SMILES = fill("", num_rows_raw_params),
        Mw = fill(0.0, num_rows_raw_params),
        m = fill([0.0], num_rows_raw_params), 
        σ = fill([0.0], num_rows_raw_params),
        λ_a = fill([0.0], num_rows_raw_params),
        λ_r = fill([0.0], num_rows_raw_params),
        ϵ = fill([0.0], num_rows_raw_params),
        split = fill("", num_rows_raw_params)
    ) ;

    processed_data.species = unique_values_raw_params;

    num_epochs_plotting = 0
    vmin = 0
    vmax = 0

    for i in unique_values_raw_params
        raw_params_spec = filter(row -> isequal(row.species, i), raw_params)
        num_epochs_plotting = nrow(raw_params_spec)

        row_number_processed = findall(isequal(i), processed_data.species)
        row_number_source = findall(isequal(i), pcp_source_params.common_name)

        processed_data[row_number_processed,:isomeric_SMILES] = pcp_source_params[row_number_source,:isomeric_SMILES]
        processed_data[row_number_processed,:Mw] = [raw_params_spec[1,:Mw]]
        processed_data[row_number_processed,:split] = [raw_params_spec[1,:split]]

        if start_row_interval ≠ 0 && end_row_interval ≠ 0
            processed_data[row_number_processed,:m] = [[raw_params_spec[i,:m] for i = start_row_interval:end_row_interval]]
            processed_data[row_number_processed,:σ] = [[raw_params_spec[i,:σ] for i = start_row_interval:end_row_interval]]
            processed_data[row_number_processed,:ϵ] = [[raw_params_spec[i,:ϵ] for i = start_row_interval:end_row_interval]]
            processed_data[row_number_processed,:λ_a] = [[raw_params_spec[i,:λ_a] for i = start_row_interval:end_row_interval]]
            processed_data[row_number_processed,:λ_r] = [[raw_params_spec[i,:λ_r] for i = start_row_interval:end_row_interval]]
            num_epochs_plotting = end_row_interval - start_row_interval + 1
            vmin = start_row_interval
            vmax = end_row_interval

        elseif last_n_rows ≠ 0
            processed_data[row_number_processed,:m] = [[raw_params_spec[i,:m] for i = num_epochs_plotting - last_n_rows + 1:num_epochs_plotting]]
            processed_data[row_number_processed,:σ] = [[raw_params_spec[i,:σ] for i = num_epochs_plotting - last_n_rows + 1:num_epochs_plotting]]
            processed_data[row_number_processed,:ϵ] = [[raw_params_spec[i,:ϵ] for i = num_epochs_plotting - last_n_rows + 1:num_epochs_plotting]]
            processed_data[row_number_processed,:λ_a] = [[raw_params_spec[i,:λ_a] for i = num_epochs_plotting - last_n_rows + 1:num_epochs_plotting]]
            processed_data[row_number_processed,:λ_r] = [[raw_params_spec[i,:λ_r] for i = num_epochs_plotting - last_n_rows + 1:num_epochs_plotting]]
            vmin = num_epochs_plotting - last_n_rows + 1
            vmax = num_epochs_plotting

            num_epochs_plotting = last_n_rows

        else
            processed_data[row_number_processed,:m] = [[raw_params_spec[i,:m] for i = 1:num_epochs_plotting]]
            processed_data[row_number_processed,:σ] = [[raw_params_spec[i,:σ] for i = 1:num_epochs_plotting]]
            processed_data[row_number_processed,:ϵ] = [[raw_params_spec[i,:ϵ] for i = 1:num_epochs_plotting]]
            processed_data[row_number_processed,:λ_a] = [[raw_params_spec[i,:λ_a] for i = 1:num_epochs_plotting]]
            processed_data[row_number_processed,:λ_r] = [[raw_params_spec[i,:λ_r] for i = 1:num_epochs_plotting]]
            vmin = 1
            vmax = num_epochs_plotting
        end
        
    end

    processed_data = sort(processed_data, :Mw)
    
    return processed_data
end;

function sat_props_calc_VrMie(; species, processed_data_split, pcp_source_params_split,plot_all_exptl_data = true,epoch=1,second_deriv_props_p=10^5,n_points=2000,same_temp_range=false)
    
    v_liq_range_vrmie = []
    v_vap_range_vrmie = []
    p_range_vrmie = []
    cp_range_vrmie = []        

    processed_data_test_species = filter(row -> row.species == lowercase(species), processed_data_split);
    source_data_test_species = filter(row -> row.common_name == lowercase(species), pcp_source_params_split);
    
    pcp_model = PPCSAFT([species])
    Tc_pcp, pc_pcp, Vc_pcp = crit_pure(pcp_model)

    # Create SAFT-VR Mie model
    Mw_test_species = processed_data_test_species[1,:Mw]
    m_test_species = processed_data_test_species[1,:m][epoch]
    σ_test_species = processed_data_test_species[1,:σ][epoch]
    λ_a_test_species = processed_data_test_species[1,:λ_a][epoch]
    λ_r_test_species = processed_data_test_species[1,:λ_r][epoch]
    ϵ_test_species = processed_data_test_species[1,:ϵ][epoch]
    vrmie_model = make_model(Mw_test_species, m_test_species, σ_test_species, λ_a_test_species, λ_r_test_species, ϵ_test_species)
            
    # compute predicted data
    Tc0 = BigFloat.(Clapeyron.x0_crit_pure(vrmie_model))
    options = Clapeyron.NEqOptions(maxiter=20_000)
    Tc_vrmie, pc_vrmie, Vc_vrmie = Float64.(crit_pure(vrmie_model, Tc0; options=options))
            
    # set experimental data range
    if plot_all_exptl_data == false
        Tmin = source_data_test_species.expt_T_min_liberal[1]

        if same_temp_range == false
            Tmax = Tc_vrmie
        else
            Tmax = min(500, 0.95*Tc_pcp,source_data_test_species.expt_T_max_conservative[1])
        end

        T_range = collect(range(Tmin, Tmax, n_points))
    else
        Tmin = min(source_data_test_species.expt_T_min_liberal[1],0.5 * Tc_pcp)

        if same_temp_range == false
            Tmax = Tc_vrmie
        else
            Tmax = Tc_pcp
        end

        T_range = collect(range(Tmin, Tmax, n_points))
    end

    # compute predicted data
    for T in T_range

        if isempty(v_liq_range_vrmie)
            method = ChemPotVSaturation()
        else
            method = ChemPotVSaturation(vl=last(v_liq_range_vrmie), vv=last(v_vap_range_vrmie))
        end

        (p_sat, v_liq_sat, v_vap_sat) = saturation_pressure(vrmie_model, T, method)
        cp_vrmie = isobaric_heat_capacity(vrmie_model, second_deriv_props_p, T)

        push!(v_liq_range_vrmie, v_liq_sat)
        push!(v_vap_range_vrmie, v_vap_sat)
        push!(p_range_vrmie, p_sat)
        push!(cp_range_vrmie, cp_vrmie)
    end 

    return T_range, [Tc_vrmie, pc_vrmie, Vc_vrmie], v_liq_range_vrmie, v_vap_range_vrmie, p_range_vrmie, cp_range_vrmie

end

function sat_props_calc_PCP(; species, source_data_test_species, plot_all_exptl_data = true,second_deriv_props_p=10^5,n_points = 50,same_temp_range=false)

    v_liq_range_pcp = []
    v_vap_range_pcp = []
    p_range_pcp = []
    cp_range_pcp = []     
    
    # Create PCP-SAFT model and compute critical props
    pcp_model = PPCSAFT([species])
    Tc_pcp, pc_pcp, Vc_pcp = crit_pure(pcp_model)
    
    # set experimental data range
    if plot_all_exptl_data == false
        Tmin = source_data_test_species.expt_T_min_liberal[1]
        Tmax = min(500, 0.95*Tc_pcp,source_data_test_species.expt_T_max_conservative[1])
        T_range = collect(range(Tmin, Tmax, n_points))
    else
        Tmin = min(source_data_test_species.expt_T_min_liberal[1],0.5 * Tc_pcp)
        Tmax = Tc_pcp
        T_range = collect(range(Tmin, Tmax, n_points))
    end
    
    # compute pseudo-experimental data
    for T in T_range
        
        (p_sat, v_liq_sat, v_vap_sat) = saturation_pressure(pcp_model, T)
        cp_pcp = isobaric_heat_capacity(pcp_model, second_deriv_props_p, T)

        push!(v_liq_range_pcp, v_liq_sat)
        push!(v_vap_range_pcp, v_vap_sat)
        push!(p_range_pcp,p_sat)
        push!(cp_range_pcp,cp_pcp)
    end

    return T_range, [Tc_pcp, pc_pcp, Vc_pcp], v_liq_range_pcp, v_vap_range_pcp, p_range_pcp, cp_range_pcp

end;

function find_key(dict, target_str)
    for (key, values) in dict
        if target_str in values
            return key
        end
    end
    return nothing
end;

function remove_nans(pcp_prop, vrmie_prop)
    
    nan_indices_pcp = findall(isnan, pcp_prop)
    vrmie_prop = vrmie_prop[setdiff(1:length(vrmie_prop), nan_indices_pcp)]
    pcp_prop = pcp_prop[setdiff(1:length(pcp_prop), nan_indices_pcp)]

    nan_indices_vrmie = findall(isnan, vrmie_prop)
    vrmie_prop = vrmie_prop[setdiff(1:length(vrmie_prop), nan_indices_vrmie)]
    pcp_prop = pcp_prop[setdiff(1:length(pcp_prop), nan_indices_vrmie)]

    return pcp_prop, vrmie_prop
end;

function readout_file_analysis(; files_for_val_error, line_start_main_training, files_to_average)

    all_train_losses_main = []
    all_val_losses_main = []
    epoch_ranges_main = []
    epoch_times_main = []

    lines_files_main = readlines.(files_for_val_error)

    for lines in lines_files_main

        epoch = 1
        training_batch_losses = []
        validation_batch_losses = []
        epoch_times = []

        for line in lines[line_start_main_training:end]

            parts = split(line,"abc")

            for part in parts 

                if occursin("epoch $epoch:", part)

                    training_batch_loss = parse(Float64,(split(split(part, ",")[1], "= ")[2]))
                    validation_batch_loss = parse(Float64,(split(split(part, ",")[2], "val_loss = ")[2]))
                    epoch_time = parse(Float64,(split(split(split(part, ",")[3], "= ")[2], "s")[1]))

                    push!(training_batch_losses, Float64(training_batch_loss))
                    push!(validation_batch_losses, Float64(validation_batch_loss))
                    push!(epoch_times, epoch_time)

                    epoch += 1
                end

            end
        end

        epoch_range = collect(range(1,epoch-1))
        push!(all_val_losses_main, validation_batch_losses)
        push!(all_train_losses_main, training_batch_losses)
        push!(epoch_ranges_main, epoch_range)
        push!(epoch_times_main, epoch_times)

    end

    total_epochs = length.(epoch_ranges_main)
    shortest_total_epochs = minimum([total_epochs[i] for i in files_to_average])

    println(total_epochs)

    average_train_loss = [mean([all_train_losses_main[i][j] for i in files_to_average]) for j = 1:shortest_total_epochs]
    average_val_loss = [mean([all_val_losses_main[i][j] for i in files_to_average]) for j = 1:shortest_total_epochs]
    average_epoch_time = [mean([epoch_times_main[i][j] for i in files_to_average]) for j = 1:shortest_total_epochs]

    epochs_min_val_loss = argmin(abs.(average_val_loss .- minimum(average_val_loss)))

    return epochs_min_val_loss, total_epochs, average_val_loss, average_train_loss, average_epoch_time

end;