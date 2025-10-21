# To change this template, choose Tools | Templates
# and open the template in the editor.

__author__="tmfilho"
__date__ ="$30/12/2011 11:15:29$"

from random import shuffle
from random import randint
from random import seed
from math import sqrt
import sys
import math
from numpy import zeros
from util import geraNormaisMultiVariadas

def lerDados(nome):
    classes = []
    with open(nome, 'r') as f:
        for line in f:
            linha = []
            variaveis = line.split()
            for i,v in enumerate(variaveis):
                if i != (len(variaveis) - 1):
                    linha.append(float(v))
                else:
                    linha.append(int(v))
            if linha[len(linha)-1] == len(classes):
                classes.append([])
            classes[linha[len(linha)-1]].append(linha)
    return classes

def gerarInstancia(parametros,classe):
        #calcula os valores uniformes das duas variaveis
        valor = randint(-sys.maxint-1,sys.maxint)%100
        u1 = valor/100.0
        valor = randint(-sys.maxint-1,sys.maxint)%100
        u2 = valor/100.0
        #para ter certeza que u2 e diferente de zero
        while u2 <= 0.0:
            valor = randint(-sys.maxint-1,sys.maxint)%100
            u2 = valor/100.0
        expo = -math.log(u2)
        r = expo * 2
        expo = math.sqrt(r)
        #temp e a primeira variavel vezes duas vezes pi
        temp = 2*math.pi*u1
        #calcula as exponencias das duas variaveis
        z1 = expo*math.cos(temp)
        z2 = expo*math.sin(temp)
        #calculando o valor da segunda variavel do elemento j
        x2 = parametros[classe][2]+(parametros[classe][3]*z2)
        #dependendo do ro, a primeira variavel pode assumir valor diferente pois X1 e condicional a X2
        #portanto, calcularemos a media condicional da primeira variavel
        mediaCond = parametros[classe][0]+(((parametros[classe][4]*parametros[classe][1])/parametros[classe][3])*(x2-parametros[classe][2]))
        #ro da classe i ao quadrado
        roQua = pow(parametros[classe][4], 2)
        #1 - ro ao quadrado que dara um numero entre 0 e 1
        roQua = 1-roQua
        #agora calcularemos o desvio-padrao condicional
        desvioCond = parametros[classe][1]*z1*pow(roQua, 0.5)
        #calculando o valor da primeira variavel de j
        x1 = mediaCond+desvioCond
        return [x1,x2]

def geraNormaisBivariadas(numeroClasses,parametros,intervalo):
    classes = [];
    for supremo in range(numeroClasses):
        classes.append([])
    #for que ira varrer cada classe para criar os valores dos seus elementos
    #onde i representa o numero da classe sendo avaliada no momento
    for w,v in enumerate(parametros):
        i = 0
        if w in [3,4,5]:
            i = 1
        elif w in [6,7,8,9]:
            i = 2
        elif w in [10,11,12,13,14]:
            i = 3
        #for que ira escrever os valores de cada elemento da classe
        #onde j representa o elemento que esta sendo calculado
        total = parametros[w][5]
        j = 0
        while j < total:
            vars = gerarInstancia(parametros, w)
            delta = randint(1,intervalo)
            var1 = [vars[0]-(delta/2),vars[0]+(delta/2)]
            var2 = [vars[1]-(delta/2),vars[1]+(delta/2)]
            vars = []
            vars.extend(var1)
            vars.extend(var2)
            vars.append(i)
            if not vars in classes[i]:
                classes[i].append(vars)
                j = j + 1
#    for c in classes:
#        for linha in c:
#            print linha
#    exit()
    return classes

def separarHoldOut(numeroClasses, classes, percentagem):
        treinamento = []
	teste = []
        validacao = []
        for i in range(numeroClasses):
            shuffle(classes[i])
            total = (len(classes[i])*percentagem)/100
            treinamento.extend(classes[i][0:total])
            teste.extend(classes[i][total:len(classes[i])])
            validacao = teste[0:len(teste)/2]
            teste = teste[len(teste)/2:len(teste)]
        return [treinamento,teste,validacao]

def randomizar(classes):
    classesRandomizadas = []
    for supremo in range(len(classes)):
        classesRandomizadas.append([])
        for v in classes[supremo]:
            classesRandomizadas[supremo].append(v[:])
        shuffle(classesRandomizadas[supremo])
    return classesRandomizadas

def separarFolds(classesRandomizadas,quantidadePartes):
    folds = []
    temp = []
    for v in classesRandomizadas:
        temp.extend(v[:])
    for j in range(quantidadePartes):
        folds.append([])
    i = 0
    while len(temp) > 0:
        folds[i%quantidadePartes].append(temp.pop()[:])
        i = i + 1
    return folds

def separarConjuntos(folds,quantidadePartes,parteTeste):
    treinamento = []
    teste = []
    for i in range(quantidadePartes):
        if i == parteTeste:
            for v in folds[i]:
                teste.append(v[:])
        else:
            for v in folds[i]:
                treinamento.append(v[:])
    conjuntos = [treinamento,teste]
    return conjuntos

def calcularDistancia(dado,prot):
        # compute per-original logic: squared diffs, then sum sqrt of even-indexed and odd-indexed groups
        parcelas = [pow(x - y, 2.0) for x, y in zip(dado[:len(dado) - 1], prot[:len(prot) - 1])]
        parte_even = pow(sum([x for i, x in enumerate(parcelas) if i % 2 == 0]), 0.5)
        parte_odd = pow(sum([x for i, x in enumerate(parcelas) if i % 2 != 0]), 0.5)
        return parte_even + parte_odd

def checarDelta(dado,classe):
    return 1.0 if dado[-1] == classe else 0.0

def testar(treinamento,teste,k,numeroClasses):
    """
    Weighted inverse-distance k-NN.
    - treinamento: list of training instances (features... , class)
    - teste: list of test instances (features... , class)
    - k: number of neighbors (int or float convertible to int)
    - numeroClasses: number of classes
    Returns classification error percentage on the teste set.
    """
    numeroErros = 0.0
    k = int(k)
    for ds in teste:
        # list of tuples: (index_in_training, distance, weight)
        dists = [
            (idx, distancia, 1.0 / distancia)
            for idx, v in enumerate(treinamento)
            for distancia in [calcularDistancia(ds, v)]
            if distancia > 0
        ]
        dists.sort(key=lambda t: t[1])
        kVizinhos = dists[:k]
        omegas = [
            sum([tup[2] * checarDelta(treinamento[tup[0]], classe) for tup in kVizinhos])
            for classe in range(numeroClasses)
        ]
        vencedora = omegas.index(max(omegas))
        if ds[-1] != vencedora:
            numeroErros += 1
    return (numeroErros / len(teste)) * 100.0

#    erros = experimentosAnteriores[1]
    seed(1)
    ks = []
    for j in range(monteCarlo):
        classesRandomizadas = randomizar(classes)
        folds = separarFolds(classesRandomizadas,quantidadePartes)
        for parteTeste in range(quantidadePartes):
            print "{0}%".format(parteTeste + j*quantidadePartes)
#            if (parteTeste + j*quantidadePartes) > repeticoesAnteriores[len(repeticoesAnteriores)-1]:
            conjuntos = separarConjuntos(folds,quantidadePartes, parteTeste)
            treinamento = conjuntos[0]
            if len(ks) == 0:
                ks = [[] for i in treinamento]
            teste = conjuntos[1]
            for k in range(int(sqrt(len(treinamento)))):
                erro = testar(treinamento,teste,float(k+1),numeroClasses)
                ks[k].append(erro)
#                erros.append(erro)
#                registrarUltimoExperimento(nomeArquivoResultado, parteTeste + j*quantidadePartes, erro)
    errosMedios = []
    for erros in ks:
        if len(erros) > 0:
            errosMedios.append(sum(erros)/len(erros))
        else:
            errosMedios.append(101.0)
    erroMedio = min(errosMedios)
    indice = errosMedios.index(erroMedio)
    print ks[indice]
    desvioErro = sqrt((sum([pow(x-erroMedio,2.0) for x in ks[indice]]))/(len(ks[indice])-1))
    print "k: {0} erro medio: {1} desvio: {2}".format(indice + 1, erroMedio,desvioErro)

def rodarSimulacao():
    nome = "simulados_2-1"
    print nome
    numeroClasses = 2
    monteCarlo = 100
    for intervalo in [10]:
        intervaloString = "intervalo {0}".format(intervalo)
        print intervaloString
        erros = []
        seed(1)
        ks = []
        for j in range(monteCarlo):
            print "{0}%".format(j)
            classes = geraNormaisMultiVariadas([[99,9,99,169,200,0],[108,9,99,169,200,1]],intervalo)
#            classes = geraNormaisBivariadas(numeroClasses, [[139,16,98,30,0,25],[132,4,-43,16,0,20],[47,30,35,16,0,30],[291,16,161,16,0,35],[185,24,231,22,0,15],[402,19,173,25,0,25],[170,28,-70,17,0,40],[450,21,-10,22,0,35],[300,24,-50,4,0,30],[389,9,-70,30,0,25],[360,23,0,35,0,45],[228,39,23,12,0,40],[421,22,124,19,0,50],[304,27,116,4,0,40],[200,16,127,39,0,30]], intervalo)
            conjuntos = separarHoldOut(numeroClasses, classes, 50)
            treinamento = conjuntos[0]
            teste = conjuntos[1]
            if len(ks) == 0:
                ks = [[] for i in treinamento]
            for k in range(int(sqrt(len(treinamento)))):
                erro = testar(treinamento,teste,float(k+1),numeroClasses)
                ks[k].append(erro)
        errosMedios = []
        for erros in ks:
            if len(erros) > 0:
                errosMedios.append(sum(erros)/len(erros))
            else:
                errosMedios.append(101.0)
        erroMedio = min(errosMedios)
        indice = errosMedios.index(erroMedio)
        desvioErro = sqrt((sum([pow(x-erroMedio,2.0) for x in ks[indice]]))/(len(ks[indice])-1))
        print ks[indice]
        print "k: {0} erro medio: {1} desvio: {2}".format(indice + 1, erroMedio,desvioErro)

def rodarLOOMarcus():
    nome = "base_marcus.txt"
    print nome
    classes = lerDados(nome)
    numeroClasses = len(classes)
    quantidadePartes = 50
    k = 5
    matriz = zeros((numeroClasses,numeroClasses))
    seed(1)
    classesRandomizadas = randomizar(classes)
    folds = separarFolds(classesRandomizadas,quantidadePartes)
    for parteTeste in range(quantidadePartes):
        conjuntos = separarConjuntos(folds,quantidadePartes, parteTeste)
        treinamento = conjuntos[0]
        teste = conjuntos[1]
        [original,obtida] = testarMarcus(treinamento,teste,float(k),numeroClasses)
        if original == obtida:
            matriz[original,original] = matriz[original,original] + 1
        else:
            matriz[obtida,original] = matriz[obtida,original] + 1
    print matriz

if __name__ == "__main__":
    rodarBaseReal()
    #rodarSimulacao()
    #rodarLOOMarcus()