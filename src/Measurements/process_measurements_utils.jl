#######################
## UTILITY FUNCTIONS ##
#######################

# load relevant model summary info
function load_model_summary(folder::String)

    # make sure valid folder
    @assert isdir(folder)

    # load full model summary
    model_summary = TOML.parsefile(joinpath(folder, "model_summary.toml"))
    
    # get the inverse temperature
    β = model_summary["beta"]

    # get discretization in imaginary time
    Δτ = model_summary["dtau"]

    # get length of imaginary time axis
    Lτ = model_summary["L_tau"]

    # lattice geometry info table
    geometry_info = model_summary["Geometry"]

    # construct unit cell
    unit_cell_info = geometry_info["UnitCell"]
    unit_cell = UnitCell(
        lattice_vecs = collect(unit_cell_info["LatticeVectors"][key] for key in keys(unit_cell_info["LatticeVectors"])),
        basis_vecs = collect(basis_vec_dict["r"] for basis_vec_dict in unit_cell_info["BasisVectors"]),
    )

    # construct lattice
    lattice_info = geometry_info["Lattice"]
    lattice = Lattice(
        L = lattice_info["L"],
        periodic = lattice_info["periodic"]
    )

    # initialize model geometry struct
    model_geometry = ModelGeometry(unit_cell, lattice)

    # add all bond definitions to model geometry
    for bond_info in geometry_info["Bond"]

        # define bond
        bond = Bond(
            orbitals = bond_info["orbitals"],
            displacement = bond_info["displacement"]
        )

        # add bond to model geometry
        bond_id = add_bond!(model_geometry, bond)
    end

    return β, Δτ, Lτ, model_geometry
end


# get the number of MPI walkers that were simulated
function get_num_walkers(folder::String)

    # initialize the number of processes to zero
    N_process = 0

    # iterate over files in global measurement directory
    for file in readdir(joinpath(folder,"global"))

        # split file name
        atoms = split(file, ('.','-','_'))

        # get process id
        pID = parse(Int, atoms[4])

        # record the max process id
        N_process = max(N_process, pID + 1)
    end

    return N_process
end


# calculate the file intervals for each bin
function get_bin_intervals(folder::String, N_bins::Int, pID::Int = 0)

    # read in binnary data files for specified pID
    ending = @sprintf("*_pID-%d.jld2", pID)
    directory = joinpath(folder, "global")
    files = glob(ending, directory)

    # get the number of binary files
    N_files = length(files)

    # calculate the size of each bin
    N_binsize = div(N_files, N_bins)
    @assert N_files == (N_bins * N_binsize) "[N_files = $N_files] = ([N_bins = $N_bins] * [N_binsize = $N_binsize])"

    # calculate the file interval associated with each bin
    bin_intervals = UnitRange{Int}[]
    for n_bin in 1:N_bins
        bin_interval = (n_bin-1)*N_binsize+1 : n_bin*N_binsize
        push!(bin_intervals, bin_interval)
    end

    return bin_intervals
end


# calculate the average sign for each bin for a specified pID walker
function get_average_sign(folder::String, bin_intervals::Vector{UnitRange{Int}}, pID::Int)

    # get folder containing global data
    global_folder = joinpath(folder, "global")

    # get the filename for a sample binary file
    binary_file = joinpath(global_folder, @sprintf("bin-1_pID-%d.jld2", pID))

    # load sign from binary file
    sample_sign = JLD2.load(binary_file, "sgn")

    # get the data type for the sgn data
    T = typeof(sample_sign)

    # allocate array to hold binned sign data
    sgn = zeros(T, length(bin_intervals))

    # get the bin size
    N_binsize = length(bin_intervals[1])

    # get number of bins
    N_bins = length(bin_intervals)

    # iterate over bins
    for bin in 1:N_bins

        # iterate overs files in bin
        for i in bin_intervals[bin]

            # get the filename for a sample binary file
            binary_file = joinpath(global_folder, @sprintf("bin-%d_pID-%d.jld2", i, pID))

            # load sign from binary data
            sgn[bin] += JLD2.load(binary_file, "sgn")
        end

        # normalize binned sign by bin size
        sgn[bin] /= N_binsize
    end

    return sgn
end