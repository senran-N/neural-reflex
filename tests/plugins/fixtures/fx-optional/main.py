import neural_reflex as nr

nr.log("features " + ",".join(nr.features()))
nr.ready()
while not nr.wait_stop(0.1):
    pass
