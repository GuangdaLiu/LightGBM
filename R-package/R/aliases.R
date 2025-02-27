# Central location for parameter aliases.
# See https://lightgbm.readthedocs.io/en/latest/Parameters.html#core-parameters

# [description] List of respected parameter aliases specific to lgb.Dataset. Wrapped in a function to
#               take advantage of lazy evaluation (so it doesn't matter what order
#               R sources files during installation).
# [return] A named list, where each key is a parameter relevant to lgb.Dataset and each value is a character
#          vector of corresponding aliases.
.DATASET_PARAMETERS <- function() {
    all_aliases <- .PARAMETER_ALIASES()
    return(all_aliases[c(
        "bin_construct_sample_cnt"
        , "categorical_feature"
        , "data_random_seed"
        , "enable_bundle"
        , "feature_pre_filter"
        , "forcedbins_filename"
        , "group_column"
        , "header"
        , "ignore_column"
        , "is_enable_sparse"
        , "label_column"
        , "linear_tree"
        , "max_bin"
        , "max_bin_by_feature"
        , "min_data_in_bin"
        , "pre_partition"
        , "precise_float_parser"
        , "two_round"
        , "use_missing"
        , "weight_column"
        , "zero_as_missing"
    )])
}

# [description] List of respected parameter aliases. Wrapped in a function to take advantage of
#               lazy evaluation (so it doesn't matter what order R sources files during installation).
# [return] A named list, where each key is a main LightGBM parameter and each value is a character
#          vector of corresponding aliases.
.PARAMETER_ALIASES <- function() {
    params_to_aliases <- jsonlite::fromJSON(
        .Call(
            LGBM_DumpParamAliases_R
        )
    )
    for (main_name in names(params_to_aliases)) {
        aliases_with_main_name <- c(main_name, unlist(params_to_aliases[[main_name]]))
        params_to_aliases[[main_name]] <- aliases_with_main_name
    }
    return(params_to_aliases)
}

# [description]
#     Per https://github.com/microsoft/LightGBM/blob/master/docs/Parameters.rst#metric,
#     a few different strings can be used to indicate "no metrics".
# [returns]
#     A character vector
.NO_METRIC_STRINGS <- function() {
    return(
        c(
            "na"
            , "None"
            , "null"
            , "custom"
        )
    )
}
