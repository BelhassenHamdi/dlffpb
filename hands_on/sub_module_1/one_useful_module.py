class hello:
    def __init__(self):
        self.att1 = 128
        self.att2 = 'one good sting to print'

    def a_very_useful_method(self, times):
        for i in range(times):
            print('att1 is : {}, att2 is : {}'.format(self.att1, self.att2))