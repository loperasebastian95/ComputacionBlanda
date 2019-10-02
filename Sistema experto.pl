:- use_module(library(pce)).
:- use_module(library(pce_style_item)).
main:-
new(Menu, dialog('Sistema experto de hospital', size(500,500))),
new(L, label(nombre, 'Bienvenido a su diagnostico')),
new(Z, label(nombre, 'Doctor Sebastian Lopera')),
new(Y, label(nombre, 'Doctor Luis David Restrepo Cadavid')),
new(X, label(nombre, 'Doctor Edwin Enciso')),




new(@texto, label(nombre, 'Segun la respuestas dadas tendra su resultado:')),
new(@respl, label(nombre, '')),
new(Salir, button('Salir', and(message(Menu,destroy), message(Menu, free)))),
new(@boton, button('Realizar test', message(@prolog, botones))),
send(Menu,append(L)), new(@btncarrera, button('�Diagnotico?')),
send(Menu,display,L,point(100,20)),
send(Menu,display,Z,point(100,380)),
send(Menu,display,Y,point(100,360)),
send(Menu,display,X,point(100,340)),
send(Menu,display,@boton,point(130,150)),
send(Menu,display,@texto,point(50,100)),
send(Menu,display,Salir,point(20,400)),
send(Menu,display,@respl,point(20,130)),
send(Menu,open_centered).

enfermedades('COLESTEROL:

TRATAMIENTO:

medicamentos
una dieta saludable
ejercicio.
'):- colesterol,!.
enfermedades('DIABETES:
        TRATAMIENTO:

Alimentaci�n saludable
Actividad f�sica
Insulina
Medicamentos orales u otros
Trasplante '):- diabetes,!.
enfermedades('EBOLA:
        TRATAMIENTO:

L�quidos administrados por v�a intravenosa
Ox�geno
Manejo de la presi�n arterial
Medicamentos orales u otros'):-ebola,!.
enfermedades('GASTRITIS:
        TRATAMIENTO:

Anti�cido
Inhibidor de la bomba de protones
Antibi�tico de penicilina
Antidiarreicos'):-gastritis,!.
enfermedades('NEUMONIA:
        TRATAMIENTO:

Antibi�tico
Antibi�tico de penicilina
Terapia de rehidrataci�n oral
L�quidos intravenosos'):-neumonia,!.
enfermedades('PARKINSON:
        TRATAMIENTO:
Carbidopa-levodopa
Infusi�n de carbidopa-levodopa
Agonistas de la dopamina
Inhibidores de la enzima monoamino oxidasa tipo B (MAO-B)
Inhibidores de la catecol-O-metiltransferasa (COMT)
Anticolin�rgicos
L�quidos intravenosos'):-parkinson,!.
enfermedades('No estoy entrenado para darte ese diagnostico').


colesterol :-
tiene_colesterol,
pregunta('�Tiene hinchazon en alguna extremidad?'),
pregunta('Tiene perdida del equilibrio?'),
pregunta('Tiene dolor de cabeza?'),
pregunta('Tiene amarillos los ojos?'),
pregunta('Tiene vision borrosa?'),
pregunta('Tiene  agitacion,en especial al caminar o al realizar actividades leve?'),
pregunta('Tiene dolor en el pecho?').

diabetes :-
tiene_diabetes,
pregunta('Tiene sed constante?'),
pregunta('Tiene hambre excesiva?'),
pregunta('Tiene perdida de peso inexplicable?'),
pregunta('Se siente fatigado?'),
pregunta('Tiene irritabilidad?'),
pregunta('Tiene vision borrosa?').

ebola :-
tiene_ebola,
pregunta('�Presenta dolores musculares?'),
pregunta('�Tiene v�mito y diarrea?'),
pregunta('�Presenta erupciones cutaneas?'),
pregunta('�Siente debilidad intensa?'),
pregunta('�Tiene dolor de garganta?').

gastritis :-
tiene_gastritits,
pregunta('�Tiene acidez estomacal?'),
pregunta('�Presenta aerofagia?'),
pregunta('�Tiene ausencia de hambre que en ocasiones puede producir perdida de peso?'),
pregunta('�Presenta heces de color negro o con sangrado?'),
pregunta('�Tiene n�useas?').

neumonia :-
tiene_neumonia,
pregunta('�Tiene dolores articulares?'),
pregunta('�Presenta dificultad para respirar?'),
pregunta('�Tiene fiebre?').

parkinson :-
tiene_parkinson,
pregunta('�Ha notado alg�n cambio perdida de movimiento espont�neo y autom�tico en alguna extremidad?'),
pregunta('�Tiene dolores articulares?'),
pregunta('�Ha presentado rigidez severa en alguna region muscular?'),
pregunta('Sufre de depresi�n o ha utilizado farmacos para tratar una enfermedad semejante?'),
pregunta('Presenta algun trastorno en el sue�o?').


%desconocido :- se_desconoce_enfermedad.

tiene_colesterol:- pregunta("�Tiene adormecimiento en alguna extremidad?"),!.
tiene_diabetes:- pregunta("Padece de orina frecuente?"),!.
tiene_ebola:- pregunta('�Tiene fiebre?'),!.
tiene_gastritits:-pregunta('Tiene dolor abdominal?'),!.
tiene_neumonia:- pregunta('�Ha tenido tos constate los ultimos dos dias?'),!.
tiene_parkinson:- pregunta('�Presenta temblor en alguna de las extremidades superiores del cuerpo?'),!.

:-dynamic si/1,no/1.


preguntar(Problema):-new(Di, dialog('Examen Medico')),
new(L2, label(texto,'Responde las siguientes preguntas')),
new(La, label(prob,Problema)),

new(B1,button(si,and(message(Di,return,si)))),
new(B2,button(no,and(message(Di,return,no)))),

send(Di,append(L2)),
send(Di,append(La)),
send(Di,append(B1)),
send(Di,append(B2)),

send(Di,default_button,si),
send(Di,open_centered),
get(Di,confirm,Answer),
write(Answer),send(Di,destroy),


((Answer==si)->assert(si(Problema)); assert(no(Problema)),fail).

pregunta(S):- (si(S)->true; (no(S)->fail;preguntar(S))).
limpiar:- retract(si(_)),fail.
limpiar:- retract(no(_)),fail.
limpiar.


botones :-lim,
send(@boton,free),
send(@btncarrera,free),
enfermedades(Enter),
send(@texto, selection('De acuerdo con sus respuestas,usted padece de:')),
send(@respl, selection(Enter)),
new(@boton, button('Iniciar su evaluaci�n', message(@prolog, botones))),
send(Menu,display,@boton,point(40,50)),
send(Menu,display,@btncarrera,point(20,50)),
limpiar.

lim:- send(@respl, selection('')).

limpiar2:-
send(@texto,free),
send(@respl,free),
% send(@btncarrera,free),
send(@boton,free).