# -*- coding: utf-8 -*-
"""
Created on Sat Jul  18 16:14:24 2020
@author: Rodrigo S. de Oliveira
"""

import numpy as np
from scipy import linalg

#------------------------------SUB-ROTINAS-NECESSÁRIAS------------------------#

def delta_de_kronecker(m,n): #função delta de Kronecker
    if m==n:
        funcao = int(1)
    else:
        funcao = int(0)
    return funcao

'''
A função operadores(momento_angular_total) função calcula as matrizes 
Jx, Jy, Jz para um determinado valor de momento angular total atribuído 
a variável MAT.
'''

def operador_momento_angular(MAT):
    if MAT == 0:
        MAT = 'Não existe operador momento angular total definido para J = 0.'
        print('-> -> -> ERRO NA FUNÇÃO operadores(momento_angular_total)!')
    else:
        numero_de_linhas = int(2*(abs(MAT)) + 1)
        numero_de_colunas = int(2*(abs(MAT)) + 1)
        lam = (abs(MAT))*((abs(MAT)) + 1) 
        #lam é o valor J(J+1) referente ao autovalor do operador J_quadrado.
        Jz = np.zeros((numero_de_linhas,numero_de_colunas))
        J_mais = np.zeros((numero_de_linhas,numero_de_colunas))
        J_menos = np.zeros((numero_de_linhas,numero_de_colunas))
        for i in range(numero_de_linhas):
            m = (-1)*(abs(MAT)) + i
            for k in range(numero_de_colunas):
                n = (-1)*(abs(MAT)) + k
                Jz[i,k] = (m)*(delta_de_kronecker(i,k))
                J_menos[i,k]= ((lam-(n)*(n-1))**(0.5))*(delta_de_kronecker(i,k-1))
                J_mais[i,k] = ((lam-(n)*(n+1))**(0.5))*(delta_de_kronecker(i,k+1))
        Jx = (J_mais + J_menos)/2
        Jy = (1/(2j))*(J_mais - J_menos)
        operador_momento_total = np.array([Jx,Jy,Jz])
    return operador_momento_total

'''
A funcao_particao(energia,temperatura) é uma sub-rotina necessária para as
funções funcao_energia_livre(energia,temperatura),
funcao_magnetizacao(g,J,energia,psi,temperatura) e 
funcao_entropia_magnetica(energia,temperatura).
'''
def funcao_particao(energia,temperatura): 
    kB = 0.086 # Constante de Boltzmann em meV
    if temperatura == 0:
        z = 'beta  = 1/((kB)*(T)) não pode receber temperatura nula'
        print('-> -> -> ERRO NA FUNÇÃO ')
    else:
        beta  = 1/((kB)*(temperatura))
        z = 0
        n = len(energia)
        for i in range(n):
            fator = np.exp((-beta)*(energia[i]))
            z += fator   
    return z

'''
A função funcao_energia_livre(energia,temperatura) é uma sub-rotina 
necessária para a função modelo_magnetico(valores_entrada).
'''

def funcao_energia_livre(energia,temperatura):
    kB = 0.086 # Constante de Boltzmann em meV
    sub_nets,ene_bolt, F = len(energia), (kB)*(temperatura), 0
    for i in range(sub_nets):
        Z = funcao_particao(energia[i],temperatura)
        F += (-1)*(ene_bolt)*(np.log(Z))
    return F

'''
A função campo_efetivo(M_0,lande,lam,field, fi, theta) 
calcula o campo efetivo para cada sub-rede magnética salvando 
em uma lista [bef1,bef2,...,befn], em que befi = [befxi, befyi,befzi]
ou um array M_0 é o vetor magnetização inicial de cada 
sub-rede (M_0 = [[m1x,m1y,m1z],[m2x,m2y,m2z],...[mnx,mny,mnz]] ou um array)
lande é do tipo [g_1, g_2,...gn], em que g_n é o fator de 
Landè da sub-rede 'n'. lam é uma matriz n x n, em que os elementos da diagonal
principal são os parâmetros de troca em meV da sub-rede 'i' com ela mesma,
e os outros elementos da matriz são os parâmetros de troca em meV
da sub-rede 'i' com as outras sub-redes 
(lam = [[L11,L12,...,L1n],[L21,L22,...,L2n],...[Ln1,Ln2,...,Lnn]] ou um array)
field é o campo magnético externo em Tesla
theta é o ângulo com o eixo 'x' da projeção do vetor 
campo magnético externo no plano xy
fi é o ângulo com o eixo 'z' do vetor campo magnético externo
'''
def campo_efetivo(M_0,lande,lam,campo):
    miB = 0.0578838 # Magneton de Bohr em meV/T^2
    Bx = campo[0]
    By = campo[1]
    Bz = campo[2]
    Alpha = (miB)*lande
    campo_efetivo = []
    for i in range (len(lande)):
        Bef_x = (Alpha[i])*(Bx)
        Bef_y = (Alpha[i])*(By)
        Bef_z = (Alpha[i])*(Bz)
        for j in range (len(lande)):
            Bef_x += ((lam[i,j])/(Alpha[j]))*(M_0[j][0]) 
            Bef_y += ((lam[i,j])/(Alpha[j]))*(M_0[j][1])
            Bef_z += ((lam[i,j])/(Alpha[j]))*(M_0[j][2])
            Bef = np.array([Bef_x,Bef_y,Bef_z])
        campo_efetivo.append(Bef)
    return campo_efetivo
'''
A função hamiltoniano(operador_spin,Bef,campocristalino) 
calula as energias e as autofunções do operador hamiltoniano
de cada sub-rede. Cada elemento da lista 'eigenvalues' contém
os valores de energia da sub-rede 'i'.Cada elemento da lista 
'eigenvectors' contém os autovetores de energia da sub-rede 'i'
operador_spin é o operador de spin de cada sub-rede. Ela deve 
ser uma lista do tipo 
operador_spin = [operadores(J1),operadores(J2),...,operadores(Jn)].
Bef é o campo efetivo da sub-rede calculado pela função 
campo_efetivo(M_0,lande,lam,field, fi, theta). 
Já a variável campocristalino é o campo cristalino de cada sub-rede. 
Essa variável deve ser uma lista do tipo 
operador_spin = [Hc1,Hc2,...,Hcn], em que Hci é 
uma matriz da ordem (2*(Ji) + 1)
'''

def hamiltoniano(operador_spin,Bef,campocristalino):
    eigenvalues = []
    eigenvectors = []
    for i in range(len(operador_spin)):
        Hx = - Bef[i][0]*(operador_spin[i][0])
        Hy = - Bef[i][1]*(operador_spin[i][1])
        Hz = - Bef[i][2]*(operador_spin[i][2])
        H = Hx + Hy + Hz + campocristalino[i]
        autovalores, autovetores = linalg.eigh(H)
        eigenvalues.append(autovalores)
        eigenvectors.append(autovetores)
    return eigenvalues, eigenvectors

'''
A função anisotropia_de_forma(d,spin) calcula o hamiltoniano 
de campo cristalino para uma anisotropia de forma.D é o parâmetro 
de anisotropia e spin é o valor do momento 
angular total para esse sistema.
'''
def anisotropia_de_forma(d,spin):
    if d ==0:
        Hc = 0
    else:
        J = operadores(spin)
        Jz2 = J[2]@J[2]
        Hc = -d*Jz2
    return Hc
'''
A função campo_cristalino_cubico(constantes,spin) 
calcula o campo cristalino de geometria cúbica na
notação de Lea-Leask-Wolf, para o composto de momento
angular total de valor igual a variável spin
constantes = [W, F4,F6, xsi] são as contantes definadas
no trabalho Lea-Leask-Wolf
'''
def campo_cristalino_cubico(constantes,spin):
    W, F4 = constantes[0],constantes[1]
    F6, xsi = constantes[2], constantes[3]
    if W==0:
        Hc = 0
    else:
        momento_total = operador_momento_angular(spin)
        Jx = momento_total[0]
        Jy = momento_total[1]
        Jz = momento_total[2] 
        Jmais, I = Jx + (1j)*Jy, np.identity(len(Jx))
        Jmenos = Jx - (1j)*Jy 
        J, A = spin*I,(spin + 1)*I 
        J2, J3 = J@J, J@J@J
        A2, A3 = (A)@(A), (A)@(A)@(A)
        Jz2, Jz4  = Jz@Jz, Jz@Jz@Jz@Jz
        Jz6 = Jz@Jz@Jz@Jz@Jz@Jz
        Jmais4 = Jmais@Jmais@Jmais@Jmais
        Jmenos4 = Jmenos@Jmenos@Jmenos@Jmenos
        x1 = (35)*(Jz4)
        x2 = -(((30)*(J@A) - 25*I))@Jz2
        x3 = (-6)*(J@A) 
        x4 = (3)*(J2@A2)
        O40 = x1 + x2 + x3 + x4
        O44 = (1/2)*(Jmais4 + Jmenos4)
        O4 = O40 + (5)*O44
        y1 = 231*Jz6 - 315*J@A@Jz4 + 735*Jz4 
        y2 = 105*J2@A2@Jz2 - 525*J@A@Jz2 + 294*Jz2
        y3 = -5*J3@A3 + 40*J2@A2 - 60*J@A
        O60 = y1 + y2 + y3
        z1 = (11*Jz2 - J@A - 38*I)
        z2 = (z1)@(Jmais4 + Jmenos4)
        z3 = (Jmais4 + Jmenos4)@(z1)
        O64 = (1/4)*(z2 + z3)
        O6 = O60 + (-21)*O64
        psi = (1-abs(xsi))
        Hc = W*((xsi)*(O4/F4) + (psi)*(O6/F6))
    return Hc
'''
A função campo_cristalino_hexagonal(constantes,spin) 
calcula a hamiltoniana de campo cristalino para 
uma rede de momento magnético total dos ions igual
a spin. constantes = [B02,B04,B06,B66] são as 
constantes de campo cristalino.
'''
def campo_cristalino_hexagonal(constantes,spin):
    B02, B04 = constantes[0],constantes[1]
    B06, B66 = constantes[2],constantes[3]
    momento_total = operador_momento_angular(spin)
    Jx = momento_total[0]
    Jy = momento_total[1]
    Jz = momento_total[2]
    Jmais, I = Jx + (1j)*Jy, np.identity(len(Jx))
    Jmenos = Jx - (1j)*Jy 
    J, A = spin*I,(spin + 1)*I 
    J2, J3 = J@J, J@J@J
    A2, A3 = (A)@(A), (A)@(A)@(A)
    Jz2, Jz4  = Jz@Jz, Jz@Jz@Jz@Jz
    Jz6 = Jz@Jz@Jz@Jz@Jz@Jz
    Jmais6 = Jmais@Jmais@Jmais@Jmais@Jmais@Jmais
    Jmenos6 = Jmenos@Jmenos@Jmenos@Jmenos@Jmenos@Jmenos
    O20 = (3)*Jz2 - J@A
    x1 = (35)*(Jz4)
    x2 = -(((30)*(J@A) - 25*I))@Jz2
    x3 = (-6)*(J@A) 
    x4 = (3)*(J2@A2)
    O40 = x1 + x2 + x3 + x4
    y1 = 231*Jz6 - 315*J@A@Jz4 + 735*Jz4 
    y2 = 105*J2@A2@Jz2 - 525*J@A@Jz2 + 294*Jz2
    y3 = -5*J3@A3 + 40*J2@A2 - 60*J@A
    O60 = y1 + y2 + y3
    O66 = (1/2)*(Jmais6 + Jmenos6)
    Hc = B02*O20 + B04*O40 + B06*O60 + B66*O66
    return Hc

'''
A função funcao_magnetizacao(g,J,energia,psi,temperatura)
é uma sub-rotina necessária para as funções
modelo_magnetico_temperatura(valores_entrada) e
modelo_magnetico_campo_magnetico(valores_entrada).
'''

def funcao_magnetizacao(g,J,energia,psi,temperatura):
    miB = 0.0578838 # Magneton de Bohr em meV 
    kB = 0.086 # Constante de Boltzmann em meV
    if temperatura == 0:
        MAG = 'beta  = 1/((kB)*(T)) não pode receber temperatura nula'
        print('->->-> ERRO NA FUNÇÃO ')
    else:
        M = []
        beta  = 1./((kB)*(temperatura))
        for j in range(len(g)):
            Z = funcao_particao(energia[j],temperatura)
            n = len(energia[j])
            soma_x, soma_y, soma_z = 0, 0, 0
            for i in range(n):
                vetor = psi[j][:,i]
                vetor_estrela = (vetor.transpose()).conj()
                k_x = (vetor_estrela)@(J[j][0])@(vetor)
                k_y = (vetor_estrela)@(J[j][1])@(vetor)
                k_z = (vetor_estrela)@(J[j][2])@(vetor)
                soma_x += (k_x)*(np.exp((-beta)*(energia[j][i])))
                soma_y += (k_y)*(np.exp((-beta)*(energia[j][i])))
                soma_z += (k_z)*(np.exp((-beta)*(energia[j][i])))
            magx = np.real((g[j])*(miB)*((soma_x)/Z))
            magy = np.real((g[j])*(miB)*((soma_y)/Z))
            magz = np.real((g[j])*(miB)*((soma_z)/Z))
            mag = np.array([magx,magy,magz])
            M.append(mag)
        MAG = np.array(M)
    return MAG

'''
A função funcao_entropia_magnetica(energia,temperatura) ]
é uma sub-rotina necessária para as funções
modelo_magnetico_temperatura(valores_entrada) e
modelo_magnetico_campo_magnetico(valores_entrada).
'''

def funcao_entropia_magnetica(energia,temperatura):
    kB = 0.086 # Constante de Boltzmann em meV 
    R = 8.314  # Constante universal dos gases perfeitos(J/mol.K)
    if temperatura == 0.:
        S = 'beta  = 1/((kB)*(T)) não pode receber temperatura nula'
        print('->->-> ERRO NA FUNÇÃO ')
    else:
        beta  = 1/((kB)*(temperatura)) 
        S = [] #Lista para guardar os valores de entropia para cada sub-rede
        for energy in energia: 
            n = len(energy) 
            Z = funcao_particao(energy,temperatura) 
            soma = 0
            for i in range(n):
                fator = np.exp((-beta)*(energy[i]))
                soma += energy[i]*fator 
                epsolon = soma/Z 
                Entropia = R*(np.log(Z) + (beta)*(epsolon))
            S.append(Entropia) 
    return S

from joblib import Parallel, delayed

N_JOBS = 8 # número de processo simultâneos (olhe o numero de processadores lógicos do seu pc)

'''
A função modelo_magnetico_temperatura(valores_entrada) calcula os valores de
magnetização de cada sub-rede magnética, a magnetização total e a 
magnetização na direção do campo magnético aplicado como também a
entropia de cada sub-rede e a entropia total cuja ordem de saída 
pode ser vista na própria função, de modo que a autoconsistência
é feita na temperatura (linhas). Os campos magnéticos serão as 
colunas. A entrada deve ser uma lista com os elementos 
[J,HC,g,Lambda,pesos,chute_inicial,B,fi,teta,T]
em que J é um array com os operadores momentos angulares totais
de cada sub-rede (np.array([J1,J2,...JN])), HC é um array com 
os campos cristalinos de cada sub-rede (np.array([Hc1,Hc2,...HcN]))
g é um array com os fatores de landé de cada sub-rede 
(np.array([g1,g2,...gN])), lambda é uma matriz com os parâmetros
de troca entre as sub-redes. Esta variável deve ser escrita com
np.array[[lam11,lam12,...,lam1N],[lam21,lam22,...,lam2N],...,[lamN1,lamN2,...,lamNN]]
A variável pesos é a concentração de cada sub-rede na rede total
(np.array([p1,p1,...,pN])). A variável chute_inicial é o valor
do chute inicial da magnetização para cada sub-rede e este deve ser escrito 
da seguinte forma np.array([[m1x,m1y,m1z],[m2x,m2y,m2z],...,[mNx,mNy,mNz]])
B é um array com os campos magnéticos que serão calculados os resultados
(np.array([B1,B2,B3,...])). As varíaveis fi e teta são as direções do campo
magnético aplicado. fi é o ângulo polar e teta o angulo azimutal.
T é um array com os valores de temperatura (exemplo:np.arange(1.,70.1,1.))
'''

def modelo_magnetico_temperatura(valores_entrada):
    miB = 0.0578838 # magneton de Bohr(meV/T^2).
    J = valores_entrada[0]
    Hc = valores_entrada[1]
    g = valores_entrada[2]
    lam = valores_entrada[3] 
    pesos = valores_entrada[4]
    chute_inicial = valores_entrada[5] 
    field = valores_entrada[6]  
    fi = valores_entrada[7]
    theta = valores_entrada[8]
    temp = valores_entrada[9]
    ux = (np.sin(fi))*(np.cos(theta))
    uy = (np.sin(fi))*(np.sin(theta))
    uz = (np.cos(fi))
    vec_u = np.array([ux,uy,uz]) #direção do campo aplicado
    subredes = len(g)
    linhas = len(temp)
    entropy = np.zeros(linhas)
    Mh = np.zeros(linhas)
    Mag = np.zeros(linhas)
    free_energy = np.zeros(linhas)
    Mag_sub_x, Mag_sub_y, Mag_sub_z, entropy_sub = [],[],[],[]    
    for count1 in range(subredes):
        Mag_sub_x.append(np.zeros(linhas))
        Mag_sub_y.append(np.zeros(linhas))
        Mag_sub_z.append(np.zeros(linhas))
        entropy_sub.append(np.zeros(linhas))
    repeticoes = 5000
    chute_inicial = valores_entrada[5] 
    B = field*vec_u #Vetor campo magnético.
    for j in range(linhas):
        #chute_inicial = valores_entrada[5]
        for k in range(repeticoes): #Esse for é para realizar a autoconsistência.
            Bef = campo_efetivo(chute_inicial,g,lam,B) #calcula o campo efetivo de cada sub-rede.
            ener, psi = hamiltoniano(J,Bef,Hc) #calcula as autofunções e as energias de cada sub-rede.
            mag_subnets = funcao_magnetizacao(g,J,ener,psi,temp[j]) #calcula a magnetização de cada sub-rede.
            delta = np.sum(np.abs(mag_subnets-chute_inicial))
            if delta <= 0.0001: #autoconsistência.
                ener_livre = funcao_energia_livre(ener,temp[j])
                S_sub = funcao_entropia_magnetica(ener,temp[j]) #entropia magnética de cada sub-rede
                #entropia_magnetica(ener,temp) pela constante universal dos gases perfeitos (R).
                mx, my, mz, Entropia, energia_total = 0, 0, 0, 0, 0
                for c1 in range(subredes):
                    mx += pesos[c1]*mag_subnets[c1][0] #calcula a magnetização total na direção x
                    my += pesos[c1]*mag_subnets[c1][1] #calcula a magnetização total na direção y
                    mz += pesos[c1]*mag_subnets[c1][2] #calcula a magnetização total na direção z
                    Entropia += pesos[c1]*(S_sub[c1]) #calcula a entropia total
                    Mag_sub_x[c1][j] = (1/miB)*(mag_subnets[c1][0])
                    Mag_sub_y[c1][j] = (1/miB)*(mag_subnets[c1][1])
                    Mag_sub_z[c1][j] = (1/miB)*(mag_subnets[c1][2])
                    entropy_sub[c1][j] = S_sub[c1]
                vetor_mag = (np.array([mx, my, mz])) #vetor magnetização resultante
                modulo_mag = (1/miB)*(linalg.norm(vetor_mag))# modulo do vetor magnetização resultante
                entropy[j] = Entropia
                free_energy[j] = ener_livre
                Mag[j] = modulo_mag
                if field == 0:
                    MH = modulo_mag
                else:
                    MH = (1/miB)*(np.dot(vetor_mag,vec_u)) #Magnetização projetada na direção do campo magnético aplicado
                Mh[j] = MH
                break
            else: # se a condição de delta não for satisfeita, ele faz o chute inicial ser o valor calculado em mag_subnets
            #realimentando assim o sistema
                chute_inicial = mag_subnets
    
    variaveis_saida = [None]*(8)
    variaveis_saida[0] = Mh
    variaveis_saida[1] = Mag
    variaveis_saida[2] = entropy
    variaveis_saida[3] = free_energy
    variaveis_saida[4] = Mag_sub_x
    variaveis_saida[5] = Mag_sub_y
    variaveis_saida[6] = Mag_sub_z
    variaveis_saida[7] = entropy_sub
    return variaveis_saida

import os
import shutil

diretorio = ''  # local em que está seu código

'''
from datetime import datetime
#data_hora = datetime.now()
#data_hora_formatada = data_hora.strftime('%Y_%m_%d_%H_%M_%S')
'''
composto = 'Dy_(1-x)Tb_(x)Al_(2)_test' #+ '_' + data_hora_formatada

diretorio_composto = os.path.join(diretorio, composto)

if os.path.isdir(diretorio_composto):
    shutil.rmtree(diretorio_composto) #apaga a pasta "diretorio_composto" caso ela exista
    os.mkdir(diretorio_composto) #cria a pasta "diretorio_composto" vazia
else:
    os.mkdir(diretorio_composto)
    
con = ['0.00','0.15','0.25','0.40'] #concentração de Tb

for count in con:
    pasta1 = '\\x=' + count
    new_diretorio = diretorio + composto + pasta1
    os.mkdir(new_diretorio)

#------------------------EXEMPLO_1----------------------#

miB = 0.0578838
sub_rede_1 = 'Tb'
sub_rede_2 = 'Dy'

sub_redes = [sub_rede_1,sub_rede_2]


#-----------------------------------TÉRBIO-----------------------------#

HC_TB = campo_cristalino_cubico([0.02,60,7560,0.9],6.)
J_TB = operador_momento_angular(6.)
g_TB = 3/2
DIC_FAC_TB = (miB)*(np.array([1,1,1]))
lam_TB = 0.615
mm_TB = 158.92 #Pesquisar

#-----------------------------------DY-----------------------------#

HC_DY = campo_cristalino_cubico([-0.011,60,13860,0.3],15/2)
J_DY = operador_momento_angular(15/2)
g_DY = 4/3
DIC_FAC_DY_1 = (miB)*(np.array([1,0,0]))
lam_DY = 0.261
mm_DY = 162.5

#---------APLICAÇÃO _DO_MODELO_COM_DUAS_SUB-REDES-------------------#

J = [J_DY,J_TB]
HC = [HC_DY,HC_TB]
g = np.array([g_DY,g_TB])

T = np.arange(1., 120.1, 1.)
chute_inicial = np.array([DIC_FAC_DY_1,DIC_FAC_TB])

phi = np.array([np.pi/2,np.pi/2,np.arctan(np.sqrt(2))])
theta = np.array([(np.pi)/4,0.,(np.pi)/4])

dire = ['110','100','111']

B = np.array([0.2,0.5,0.7,1.0,1.5,2.0,3.0,4.0,5.0])

deltaS = np.zeros((len(T),len(B)))
mh  = np.zeros((len(T),len(B)+1))
mag  = np.zeros((len(T),len(B)+1))
entropia  = np.zeros((len(T),len(B)+1))
E_livre  = np.zeros((len(T),len(B)+1))

lam_x_000 = np.array([[0.261,0],[0,0]])
lam_x_015 = np.array([[0.25060818,0.03825],[0.03825,0.38273478]])
lam_x_025 = np.array([[0.242888,0.05625],[0.05625,0.434871]])
lam_x_040 = np.array([[0.20216973,0.144],[0.144,0.38896015]])

lam = [lam_x_000,lam_x_015,lam_x_025,lam_x_040]
numero_sub_redes = len(sub_redes)
Mag_sub_x, Mag_sub_y, Mag_sub_z, entropy_sub = [],[],[],[]
for count1 in range(numero_sub_redes):
        Mag_sub_x.append(np.zeros((len(T),len(B)+1)))
        Mag_sub_y.append(np.zeros((len(T),len(B)+1)))
        Mag_sub_z.append(np.zeros((len(T),len(B)+1)))
        entropy_sub.append(np.zeros((len(T),len(B)+1)))

for k in range(len(con)):
    x = float(con[k])
    concentracao = np.array([(1 - x), x])
    parametro_de_troca = lam[k]
    entrada_nulo = [J, HC, g, parametro_de_troca, concentracao, chute_inicial, 0., 0., 0., T]
    saida_nulo = modelo_magnetico_temperatura(entrada_nulo)
    mh[:,0] = saida_nulo[0]
    mag[:,0] = saida_nulo[1]
    entropia[:,0] = saida_nulo[2]
    E_livre[:,0] = saida_nulo[3]
    for l in range(numero_sub_redes):
        Mag_sub_x[l][:,0] = saida_nulo[4][l]
        Mag_sub_y[l][:,0] = saida_nulo[5][l]
        Mag_sub_z[l][:,0] = saida_nulo[6][l]
        entropy_sub[l][:,0] = saida_nulo[7][l]
    pasta = os.path.join(diretorio_composto, 'x=' + con[k])
    for i in range(len(phi)):
        for j in range(len(B)):
            entrada = [J, HC, g, parametro_de_troca, concentracao, chute_inicial, B[j], phi[i], theta[i], T]
            exit_model = modelo_magnetico_temperatura(entrada)
            deltaS[:,j] = saida_nulo[2] - exit_model[2]
            mh[:,j+1] = exit_model[0]
            mag[:,j+1] = exit_model[1]
            entropia[:,j+1] = exit_model[2]
            E_livre[:,j+1] = exit_model[3]
            for l in range(numero_sub_redes):
                Mag_sub_x[l][:,j+1] = exit_model[4][l]
                Mag_sub_y[l][:,j+1] = exit_model[5][l]
                Mag_sub_z[l][:,j+1] = exit_model[6][l]
                entropy_sub[l][:,j+1] = exit_model[7][l]
        np.savetxt(os.path.join(pasta, 'delta_S_x=' + con[k] + '_direcao_' + dire[i] + '.txt'), np.c_[T, deltaS],
                   fmt='%f')
        np.savetxt(os.path.join(pasta, 'mh_x=' + con[k] + '_direcao_' + dire[i] + '.txt'), np.c_[T,mh],
                   fmt='%f')
        np.savetxt(os.path.join(pasta, 'mag_x=' + con[k] + '_direcao_' + dire[i] + '.txt'), np.c_[T, mag],
                   fmt='%f')
        np.savetxt(os.path.join(pasta, 'entropia_x=' + con[k] + '_direcao_' + dire[i] + '.txt'), np.c_[T, entropia],
                   fmt='%f')
        np.savetxt(os.path.join(pasta, 'energia_livre_x=' + con[k] + '_direcao_' + dire[i] + '.txt'),
                   np.c_[T, E_livre], fmt='%f')
        for cont in range(numero_sub_redes):
            np.savetxt(os.path.join(pasta, 'mag_x_'+ sub_redes[cont] +'_x=' + con[k] + '_direcao_' + dire[i] + '.txt'),
                   np.c_[T, Mag_sub_x[cont]], fmt='%f')
            np.savetxt(os.path.join(pasta, 'mag_y_'+ sub_redes[cont] +'_x=' + con[k] + '_direcao_' + dire[i] + '.txt'),
                   np.c_[T, Mag_sub_y[cont]], fmt='%f')
            np.savetxt(os.path.join(pasta, 'mag_z_'+ sub_redes[cont] +'_x=' + con[k] + '_direcao_' + dire[i] + '.txt'),
                   np.c_[T, Mag_sub_z[cont]], fmt='%f')
            np.savetxt(os.path.join(pasta, 'mag_z_'+ sub_redes[cont] +'_x=' + con[k] + '_direcao_' + dire[i] + '.txt'),
                   np.c_[T, entropy_sub[cont]], fmt='%f')
