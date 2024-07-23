from termcolor import cprint


cprint1 = lambda s, *args, **kwargs: cprint(s, "cyan", attrs=["bold"], *args, **kwargs)
cprintc = lambda s, *args, **kwargs: cprint(s, "cyan", *args, **kwargs)
cprintm = lambda s, *args, **kwargs: cprint(s, "magenta", *args, **kwargs)
