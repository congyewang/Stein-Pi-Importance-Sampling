from toolbox import output_gs_name
from toolbox import store_thinning


if __name__ == "__main__":
    gs_name = output_gs_name()
    for i in gs_name:
        exec(f"store_thinning(model_name='{i}')")
