import numpy as np
import random
import copy


class Gene():
    def __init__(self, lens=None, behaviors=None):
        '''
        lens: 每个性状的基因长度数组
        behaviors: 性状数组，每个性状用一个十进制数表示
        '''
        self.lens = lens
        self.gene = []
        # self.g_gene = None

        #  
        if behaviors is not None:
            for i in range(len(behaviors)):
                self.gene.append([])

            self.code(behaviors)

    def code(self, behaviors):
        '''
        convert behaviors to binary
        '''
        for i in range(len(behaviors)):
            behavior = behaviors[i]
            for j in range(self.lens[i]):
                if behavior == 0:
                    self.gene[i].append(0)
                elif behavior == 1:
                    self.gene[i].append(1)
                else:
                    self.gene[i].append(behavior % 2)
                    behavior = (behavior - self.gene[i][-1]) / 2

        self.BCG()

    def BCG(self):  
        '''
        convert binary to Gray code
        '''
        gene = copy.deepcopy(self.gene)
        for i in range(len(self.lens)):
            for j in range(self.lens[i] - 1):
                gene[i][j] = self.gene[i][j] ^ self.gene[i][j+1]
            # gene[i][self.len[i]-1] = self.gene[i][-1]

        self.gene = gene

    def GCB(self):
        gene = copy.deepcopy(self.gene)
        for i in range(len(self.lens)):
            for j in range(self.lens[i] - 2, -1, -1):
                gene[i][j] = self.gene[i][j] ^ gene[i][j + 1]

        self.gene = gene

    def decode(self):
        self.GCB()
        behavior = np.zeros(shape=(len(self.lens)), dtype=np.int)
        for i in range(len(self.lens)):
            # gene = self.gene[i]
            for j in range(self.lens[i]):
                behavior[i] += 2 ** j * self.gene[i][j]
        return behavior

    def get_behavior(self):
        return self.decode()

    def mutation(self, pm):
        gene = np.concatenate(self.gene, axis=0)

        mute_points = random.sample(range(len(gene)), int(len(gene) * pm))

        for point in mute_points:
            gene[point] = gene[point] ^ 1

        self.gene = self.recover_gene(gene)

        # return [self.recover_gene(gene), self.lens]

    def crossover(self, gene, pc):
        father_gene = np.concatenate(self.gene, axis=0)
        mother_gene = np.concatenate(gene, axis=0)

        cross_points = random.sample(range(len(father_gene)), int(len(father_gene) * pc))
        for point in cross_points:
            temp = mother_gene[point]
            mother_gene[point] = father_gene[point]
            father_gene[point] = temp

        child1 = [self.recover_gene(father_gene), self.lens]
        child2 = [self.recover_gene(mother_gene), self.lens]

        return child1, child2

    def recover_gene(self, expanded_gene):
        gene = copy.deepcopy(self.gene)
        for i in range(len(self.lens)):
            for j in range(self.lens[i]):
                gene[i][j] = gene[i*sum(expanded_gene[0:i]) + j]

        return gene

    def set_property(self, gene, lens):
        '''
        gene: 
        lens: 
        '''
        self.gene = gene
        self.lens = lens


class Xplane_Individual():
    def __init__(self, pc=0.4, pm=0.1, genelen=2, gene=None):
        self.fitness = 0
        self.Vmaps = {0:50,1:80,2:80,3:120} # th
        self.Phimaps = {0:-20,1:0,2:0,3:20} # al
        self.Thetamaps = {0:-20,1:0,2:0,3:20}   # el
        # self.Vmaps = {0: 50, 1: 80, 2: 80, 3: 120}  # th
        # self.Phimaps = {0: -20, 1: 0, 2: 0, 3: 20}  # al
        # self.Thetamaps = {0: -20, 1: 0, 2: 0, 3: 20}  # el
        self.timelen = 300
        self.gene = gene
        self.pc = pc
        self.pm = pm
        self.genelen = genelen
        # self.behavior = None

    def init_gene(self):
        behavior_abstract = np.random.randint(low=0, high=4, size=900, dtype=int)
        lens = np.ones_like(behavior_abstract) * self.genelen
        self.gene = Gene(lens=lens,behaviors=behavior_abstract)
        # self.behavior = self.recover_from_abstract(behavior_abstract)

    def recover_from_abstract(self, abstract):
        behavior = []
        behavior.append(list(map(lambda x: self.Vmaps[x], abstract[0:self.timelen])))
        behavior.append(list(map(lambda x: self.Phimaps[x], abstract[self.timelen:self.timelen*2])))
        behavior.append(list(map(lambda x: self.Thetamaps[x], abstract[self.timelen*2:self.timelen * 3])))

        return behavior

    # def convert_abstract(self, behavior):
    #     # behavior_abstract = []
    #     pass

    def get_behavior(self):
        behavior_abstract = self.gene.get_behavior()
        behavior = self.recover_from_abstract(behavior_abstract)

        return behavior, behavior_abstract

    def set_behavior_abstract(self, behavior_abstract):
        lens = np.ones_like(behavior_abstract)*2
        self.gene = Gene(lens=lens, behaviors=behavior_abstract)

    def marry(self, mate):
        child1, child2 = self.gene.crossover(mate.gene, self.pc)
        # children_gene = [Gene(), Gene()]
        gene = Gene()
        gene.set_property(gene=child1[0], lens=child1[1])
        child1_gene = copy.deepcopy(gene.mutation(self.pm))

        gene.set_property(gene=child2[0], lens=child2[1])
        child2_gene = copy.deepcopy(gene.mutation(self.pm))

        return child1_gene, child2_gene

    def set_gene(self, gene):
        # self.gene = Gene()
        # self.gene.set_property(gene, lens)
        self.gene = gene

    def set_fitness(self, fitness):
        self.fitness = fitness

class Xplane_Group():
    def __init__(self, groupSize=250, pm=0.1, pc=0.4, pn=0.1):
        self.size = groupSize
        self.pc = pc
        self.pm = pm
        self.pn = pn
        self.group = [Xplane_Individual(pc=pc, pm=pm) for i in range(groupSize)]

    def initial_group(self):
        for i in range(self.size):
            self.group[i].init_gene()

    def generate_next(self):
        self.group.sort(key=lambda x: x.fitness, reverse=True)

        next_generation = copy.deepcopy(self.group[0:int(self.size * self.pn)])

        ones = random.sample(self.group[0:int(self.size/5)], int((self.size - len(next_generation))/2))
        mates = random.sample(self.group[int(self.size/5):-1], int((self.size - len(next_generation))/2))

        for i in range(len(ones)):
            child1_gene, child2_gene = ones[i].marry(mates[i])
            next_generation.append(Xplane_Individual(pc=self.pc, pm=self.pm, genelen=2, gene=child1_gene))
            next_generation.append(Xplane_Individual(pc=self.pc, pm=self.pm, genelen=2, gene=child2_gene))

        while len(next_generation) != self.size:
            print('need {} more'.format(self.size - len(next_generation)))
            next_generation.append(self.group[self.size - len(next_generation)-1])

        self.group = next_generation

class PID_Individual():
    def __init__(self, behavior, gene_num) -> None:
        self.behavior = behavior
        self.fitness = 0
        self.gene = []

        self.len = 2
        self.gene_num = gene_num

        for i in range(self.gene_num):
            self.gene.append([])

        self.code()

    def mutate_genes(self, pm):
        for i in range(self.gene_num):
            mute_points = random.sample(range(self.len), int(self.len * pm))
            for point in mute_points:
                if self.gene[i][point] == 0.:
                    self.gene[i][point] = 1.
                else:
                    self.gene[i][point] = 0.

    def cross_genes(self, mother, pc):
        father = copy.deepcopy(self)
        for i in range(self.gene_num):
            cross_points = random.sample(range(self.len), int(self.len * pc))
            for point in cross_points:
                father.gene[i][point] = mother.gene[i][point]

        # self.fitness = 0
        return father

    def set_fitness(self, fitness):
        self.fitness = fitness

    def code(self, set=False):
        # print(self.behavior)
        if not set:
            for i in range(self.gene_num):
                behavior = np.floor(self.behavior[i] * 10000)
                for j in range(self.len):
                    if behavior == 0:
                        self.gene[i].append(0.)
                    elif behavior == 1:
                        self.gene[i].append(1.)
                        behavior = 0
                    else:
                        self.gene[i].append(behavior % 2)
                        behavior = (behavior - self.gene[i][-1]) / 2
        else:
            for i in range(self.gene_num):
                behavior = np.floor(self.behavior[i] * 10000)
                for j in range(self.len):
                    if behavior == 0:
                        self.gene[i][j] = 0.
                    elif behavior == 1:
                        self.gene[i][j] = 1.
                        behavior = 0
                    else:
                        self.gene[i][j] = behavior % 2
                        behavior = (behavior - self.gene[i][j]) / 2

    def decode(self):
        behavior = []
        for j in range(self.gene_num):
            behavior.append(0.)

        for i in range(self.gene_num):
            gene = self.gene[i]
            for j in range(self.len):
                behavior[i] += 2 ** j * gene[j]
            behavior[i] = behavior[i] / 10000

        return behavior

    def getBehavior(self):
        return self.decode()

    def setBehavior(self, behavior):
        self.behavior = behavior
        # print(self.gene)
        self.code(set=True)
        # print(self.gene)

class PID_Group():
    def __init__(self, size, behavior_num) -> None:
        self.group = []
        # self.nextGeneration = []
        self.groupSize = size
        self.pm = 0.1
        self.pc = 0.4
        self.gene_num = behavior_num

        self.generate_First()

    def generate_First(self):
        for i in range(self.groupSize):
            behavior = np.random.uniform(low=0., high=0.1024, size=(1, self.gene_num))
            self.group.append(PID_Individual(list(*behavior), self.gene_num))

    def generate_next(self):
        self.group.sort(key=lambda x: x.fitness, reverse=True)

        nextGeneration = []
        nextGeneration = copy.deepcopy(self.group[0:int(self.groupSize * 0.2)])  # natual_selection

        choices = [True, False]
        while (len(nextGeneration) != len(self.group)):
            if random.choice(choices):
                father = random.choice(self.group[0:int(self.groupSize / 5)])
            else:
                father = random.choice(self.group[int(self.groupSize / 5):int(self.groupSize / 1.5)])
            if random.choice(choices):
                mother = random.choice(self.group[0:int(self.groupSize / 5)])
            else:
                mother = random.choice(self.group[int(self.groupSize / 5):int(self.groupSize / 1.5)])

            child = father.cross_genes(mother, self.pc)
            child.mutate_genes(self.pm)

            nextGeneration.append(copy.deepcopy(child))

        self.group = nextGeneration

if __name__ == '__main__':
    # Vmaps = {0: 50, 1: 80, 2: 80, 3: 120}
    # behavior_abstract = np.random.randint(low=0, high=4, size=10, dtype=int)
    # A = list(map(lambda x: Vmaps[x], behavior_abstract[0:300]))
    # print(A)
    # print(behavior_abstract)
    print(int(5/2))
    # pass
