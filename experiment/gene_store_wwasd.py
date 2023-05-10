from toolbox import output_gs_name
from toolbox import store_wwasd


if __name__ == "__main__":
    gs_name = output_gs_name()

    for i in gs_name:
        exec(f"store_wwasd(model_name='{i}')")
