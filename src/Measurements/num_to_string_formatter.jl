function num_to_string_formatter(
    decimals::Int,
    scientific_notation::Bool
)

    fmt = scientific_notation ? "%.$(decimals)E" : "%.$(decimals)f"
    return generate_formatter(fmt)
end