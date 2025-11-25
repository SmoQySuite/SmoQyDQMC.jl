@doc raw"""
    ModelGeometry{D, T<:AbstractFloat, N}

Contains all the information defining the lattice geometry for the model in `D` spatial dimensions.

# Comment

The bond ID associated with a `bond::Bond{D}` corresponds to the index associated with it into the `bonds` vector field.

# Fields

- `unit_cell::UnitCell{D,T,N}`: Defines unit cell.
- `lattice::Lattice{D}`: Defines finite lattice extent.
- `bonds::Vector{Bond{D}}`: All available bond definitions in simulation, with vector indices giving the bond ID.
"""
struct ModelGeometry{D, T<:AbstractFloat, N}

    unit_cell::UnitCell{D,T,N}
    lattice::Lattice{D}
    bonds::Vector{Bond{D}}
end

@doc raw"""
    ModelGeometry(unit_cell::UnitCell, lattice::Lattice)

Initialize and return a [`ModelGeometry`](@ref) instance. Defines a "trivial" bond definition for each
orbital in the unit cell that connects an orbital to itself.
"""
function ModelGeometry(unit_cell::UnitCell{D}, lattice::Lattice{D}) where {D}

    # ensure all spatial dimension are periodic
    @assert all(i -> i, lattice.periodic) "All spatial dimensions in lattice must be periodic."

    # define trivial bond connecting each orbital in unit cell to itself
    n     = unit_cell.n
    bonds = Bond{D}[]
    for i in 1:n
        push!(bonds, Bond((i,i),zeros(Int,D)))
    end

    return ModelGeometry(unit_cell, lattice, bonds)
end


# print struct info in TOML format
function Base.show(io::IO, ::MIME"text/plain", model_geo::ModelGeometry{D,T}) where {D,T}

    (; unit_cell, lattice, bonds) = model_geo

    @printf io "[Geometry]\n\n"
    @printf io "dimensions = %d\n\n" D
    @printf io "[Geometry.UnitCell]\n\n"
    @printf io "orbitals = %d\n\n" unit_cell.n
    @printf io "[Geometry.UnitCell.LatticeVectors]\n\n"
    for d in 1:D
        a = @view unit_cell.lattice_vecs[:,d]
        @printf io "a_%d = %s\n" d string(round.(a, digits=6)) 
    end
    @printf io "\n"
    @printf io "[Geometry.UnitCell.ReciprocalVectors]\n\n"
    for d in 1:D
        b = @view unit_cell.reciprocal_vecs[:,d]
        @printf io "b_%d = %s\n" d string(round.(b, digits=6)) 
    end
    @printf io "\n"
    for i in 1:unit_cell.n
        r = unit_cell.basis_vecs[i]
        @printf io "[[Geometry.UnitCell.BasisVectors]]\n\n"
        @printf io "ORBITAL_ID = %d\n" i
        @printf io "r          = %s\n\n" string(round.(r, digits=6))
    end
    @printf io "\n"
    @printf io "[Geometry.Lattice]\n\n"
    @printf io "L        = %s\n" string(lattice.L)
    @printf io "periodic = [%s]\n\n" join(lattice.periodic, ", ")
    for i in eachindex(bonds)
        @printf io "[[Geometry.Bond]]\n\n"
        @printf io "BOND_ID      = %d\n" i
        @printf io "orbitals     = [%d, %d]\n" bonds[i].orbitals[1] bonds[i].orbitals[2]
        @printf io "displacement = %s\n\n" string(bonds[i].displacement)
    end

    return nothing
end


@doc raw"""
add_bond!(model_geometry::ModelGeometry{D,T}, bond::Bond{D}) where {D, T}   

Add `bond` definition to `model_geometry`, returning the bond ID i.e. the index to `bond`
in the vector `model_geometry.bonds`.
This method first checks that `bond` is not already defined. If it is this method simply
returns the corresponding bond ID. If `bond` is not already defined, then it is appended
to the vector `model_geometry.bonds`.
"""
function add_bond!(model_geometry::ModelGeometry{D,T}, bond::Bond{D}) where {D, T}

    (; bonds) = model_geometry

    # get the bond ID
    bond_id = get_bond_id(model_geometry, bond)

    # if the bond is not already recorded, then record it and get its new bond ID
    if iszero(bond_id)
        # record the bond ID
        push!(bonds, bond)
        # get the ID of the new bond
        bond_id = length(bonds)
    end

    return bond_id
end


@doc raw"""
    get_bond_id(model_geometry::ModelGeometry{D,T}, bond::Bond{D}) where {D, T}

Return the bond ID associated with the bond defintion `bond`, returning `bond_id=0`
if the it is not a recorded bond.
"""
function get_bond_id(model_geometry::ModelGeometry{D,T}, bond::Bond{D}) where {D, T}

    (; bonds) = model_geometry

    if bond in bonds
        bond_id = findfirst(b -> b==bond, bonds)
    else
        bond_id = 0
    end
    
    return bond_id
end