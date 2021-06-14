import sys

from models.vanilla_resnet import *
from models.vgg_grasp import *
from models.small_model import *


def build_model(arch, depth):
    if arch == "resnet":
        if depth == 10:
            model = resnet010()
        elif depth == 12:
            model = resnet012()
        elif depth == 14:
            model = resnet014()
        elif depth == 16:
            model = resnet016()
        elif depth == 18:
            model = resnet018()
        elif depth == 20:
            model = resnet020()
        elif depth == 22:
            model = resnet022()
        elif depth == 24:
            model = resnet024()
        elif depth == 26:
            model = resnet026()
        elif depth == 28:
            model = resnet028()
        elif depth == 30:
            model = resnet030()
        elif depth == 32:
            model = resnet032()
        elif depth == 34:
            model = resnet034()
        else:
            sys.exit("resnet doesn't implement those depth!")
    elif arch == "smresnet":
        if depth == 10:
            model = resnet010sm()
        elif depth == 12:
            model = resnet012sm()
        elif depth == 14:
            model = resnet014sm()
        elif depth == 16:
            model = resnet016sm()
        elif depth == 18:
            model = resnet018sm()
        elif depth == 20:
            model = resnet020sm()
        elif depth == 22:
            model = resnet022sm()
        elif depth == 24:
            model = resnet024sm()
        elif depth == 26:
            model = resnet026sm()
        elif depth == 28:
            model = resnet028sm()
        elif depth == 30:
            model = resnet030sm()
        elif depth == 32:
            model = resnet032sm()
        elif depth == 34:
            model = resnet034sm()
        else:
            sys.exit("smresnet doesn't implement those depth!")
    elif arch == "mdresnet":
        if depth == 10:
            model = resnet010md()
        elif depth == 12:
            model = resnet012md()
        elif depth == 14:
            model = resnet014md()
        elif depth == 16:
            model = resnet016md()
        elif depth == 18:
            model = resnet018md()
        elif depth == 20:
            model = resnet020md()
        elif depth == 22:
            model = resnet022md()
        elif depth == 24:
            model = resnet024md()
        elif depth == 26:
            model = resnet026md()
        elif depth == 28:
            model = resnet028md()
        elif depth == 30:
            model = resnet030md()
        elif depth == 32:
            model = resnet032md()
        elif depth == 34:
            model = resnet034md()
        else:
            sys.exit("mdresnet doesn't implement those depth!")
    elif arch == "vgg":
        if depth == 11:
            model = vgg11()
        elif depth == 13:
            model = vgg13()
        elif depth == 16:
            model = vgg16()
        elif depth == 19:
            model = vgg19()
        elif depth == 5:
            model = vgg5()
        elif depth == 6:
            model = vgg6()
        elif depth == 7:
            model = vgg7()
        elif depth == 51:
            model = vgg51()
        elif depth == 61:
            model = vgg61()
        elif depth == 71:
            model = vgg71()
        elif depth == 52:
            model = vgg52()
        elif depth == 62:
            model = vgg62()
        elif depth == 72:
            model = vgg72()
        else:
            sys.exit("vgg doesn't implement those depth!")

    elif arch == "small":
        if depth == 2:
            model = small2()
        elif depth == 3:
            model = small3()
        elif depth == 4:
            model = small4()
        elif depth == 5:
            model = small5()
        elif depth == 6:
            model = small6()
        elif depth == 7:
            model = small7()
        elif depth == 8:
            model = small8()
        elif depth == 9:
            model = small9()
        elif depth == 10:
            model = small10()
        else:
            sys.exit("small models doesn't implement those depth!")
    else:
        sys.exit("unknown network")

    # print(model)

    return model