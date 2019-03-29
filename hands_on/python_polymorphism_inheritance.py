class aqua_animal:
    def __init__(self, first_name='MR', last_name="Fish"):
        self.first_name = first_name
        self.last_name = last_name
        self.eyelid = True
    def swim(self):
        print('I am an aqua animal and can hold my breath eternally')
    def swim_backwards(self):
        print('not all aqua animals can swim backward')
    def skeleton(self):
        print("our squeleton is usually made of bones")

class Shark(aqua_animal):
    def __init__(self, first_name='MR', last_name="killer"):
        super(Shark, self).__init__()
        self.first_name = first_name
        self.last_name = last_name
    def swim(self):
        print("The shark is swimming.")

    def swim_backwards(self):
        print("The shark cannot swim backwards, but can sink backwards.")

    def skeleton(self):
        print("The shark's skeleton is made of cartilage.")


class Clownfish(aqua_animal):
    def swim(self):
        print("The clownfish is swimming.")

    def swim_backwards(self):
        print("The clownfish can swim backwards.")

    def skeleton(self):
        print("The clownfish's skeleton is made of bone.")

class One_cool_aqua_animal(aqua_animal):
    def swim(self):
        print("The One_cool_aqua_animal is swimming and dancing.")




sammy = Shark()
sammy.skeleton()

casey = Clownfish()
casey.skeleton()

for fish in (sammy, casey):
    fish.swim()
    fish.swim_backwards()
    fish.skeleton()
    print(fish.last_name)

dancer = One_cool_aqua_animal()
dancer.swim()
dancer.swim_backwards()
dancer.skeleton()
print(dancer.first_name)
print(dancer.eyelid)
print(sammy.eyelid)
