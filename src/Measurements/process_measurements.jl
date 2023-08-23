##############################
## PROCESS ALL MEASUREMENTS ##
##############################

function process_measurements(folder::String, N_bin::Int; time_displaced::Bool = false)

    # process global measurements
    process_global_measurements(folder, N_bin)

    # process local measurement
    process_local_measurements(folder, N_bin)

    # process correlation measurement
    process_correlation_measurements(folder, N_bin, time_displaced)

    return nothing
end


function process_measurements(folder::String, N_bin::Int, pID::Int; time_displaced::Bool = false)

    # process global measurements
    process_global_measurements(folder, N_bin, pID)

    # process local measurement
    process_local_measurements(folder, N_bin, pID)

    # process correlation measurement
    process_correlation_measurements(folder, N_bin, pID, time_displaced)

    return nothing
end