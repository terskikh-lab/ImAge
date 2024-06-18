# Description ezsave
# ================================================
# save multiple objects in a file
# =================================================
# Kenta Ninomiya @ Sanford burnham prebys medical discovery institute: 2021/08/20
# import modules=======================
import pickle


# =====================================
# @jit(nopython=True)
def ezsave(vList, file):
    with open(file, "wb") as f:
        pickle.dump(vList, f)
