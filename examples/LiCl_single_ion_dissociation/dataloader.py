"""
Load LiCl example data using pooch from figshare.

This just loads the data from the figshare repository (if not already present).
"""
import pooch

# create the pooch with the doi to the LiCl example data
DataLoaderLiCl = pooch.create(path="training_and_validation_data",
                              base_url="doi:10.6084/m9.figshare.29814989",
                              )
# and populate it from the doi
DataLoaderLiCl.load_registry_from_doi()
