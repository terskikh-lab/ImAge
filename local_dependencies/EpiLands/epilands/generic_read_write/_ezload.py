import pickle


# Description ezload
# ================================================
# load multiple objects in a file
# =================================================
# Kenta Ninomiya @ Sanford burnham prebys medical discovery institute: 2021/08/20
# @jit(nopython=True)
def ezload(file):
    with open(file, "rb") as f:
        return pickle.load(f)
