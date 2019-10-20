import math

def ackley(chromosome):
    firstSum = 0.0
    secondSum = 0.0
    for gene in chromosome:
        firstSum += gene ** 2
        secondSum += math.cos(2.0 * math.pi * gene)
    
    n = len(chromosome)
    firstExp = math.exp(-0.2 * math.sqrt(firstSum / n))
    secondExp = math.exp(secondSum / n)
    
    functionResult = -20.0 * firstExp - secondExp + 20 + math.exp(1)
    if functionResult <= 5e-15:
        functionResult = 0
    
    return functionResult

def griewank(chromosome):
    part1 = 0
    for gene in chromosome:
        part1 += gene ** 2
    
    part2 = 1
    index = 1
    for gene in chromosome:
        part2 *= math.cos(gene / math.sqrt(index))
        index += 1
        
    return 1 + part1 / 4000 - part2
