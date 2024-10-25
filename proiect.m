%% Metoda stocastic gradient descendent  
clc;
clear;
close all;

% Încarcarea setului de date
filename = 'iris.data';
opts = detectImportOptions(filename, 'FileType', 'text', 'Delimiter', ',');
opts.VariableTypes(5) = {'categorical'}; %Setarea coloanei de specii ca categorical
data = readtable(filename, opts);

% Amestecarea rândurilor pentru a evita problemele de ordonare
data = data(randperm(size(data, 1)), :);

% Extragere caracteristici și etichete
features = table2array(data(:, 1:4));
species = data.Var5; % Accesarea coloanei speciilor folosind notația cu punct

% Convertire etichete categorice în binar: 1 pentru Iris-setosa și 0 pentru celelalte
binaryLabels = double(species == 'Iris-setosa');

% Împărțirea setului de date în set de antrenament și set de testare (80%-20%)
cv = cvpartition(size(features, 1), 'HoldOut', 0.2);
idx = cv.test;

% Set de antrenament
X_train = features(~idx, :);
y_train = binaryLabels(~idx);

% Set de testare
X_test = features(idx, :);
y_test = binaryLabels(idx);

% Normalizare date de antrenament și testare
[X_train, mu, sigma] = zscore(X_train);
X_test = (X_test - mu) ./ sigma;

% Inițializare parametri rețea
m = size(X_train, 2); % Număr de caracteristici
n = 10; % Număr de neuroni în stratul ascuns
W = randn(m + 1, n) * 0.01; % Ponderi incluzând termenul de bias
x = randn(n, 1) * 0.01; % Ponderi strat de ieșire
X_train_aug = [ones(size(X_train, 1), 1), X_train]; % Date de intrare 

% Funcția de activare Aranda-Ordaz
a_param = 1;
 g = @(z) 1 - (1 + a_param * exp(z)).^(1/a_param);
g_prime = @(z) - exp(z) .* (1 + a_param * exp(z)).^(1/a_param - 1);
% Funcția de pierdere entropie încrucișată binară
cross_entropy_loss = @(y, y_pred) -mean(y .* log(y_pred) + (1 - y) .* log(1 - y_pred));

% Parametrii pentru descentul gradient stocastic
initial_learning_rate = 0.01; % Rata inițială de învățare
decay_rate = 0.9; % Factorul de descreștere a ratei de învățare
decay_steps = 1000; % Descreștere rată de învățare la fiecare 1000 de epoci
epochs = 10000;% Numărul de epoci
learning_rate = initial_learning_rate; % Rata de învățare curentă
batch_size = 1; % Mărimea batch-ului pentru SGD

% Pentru a stoca evoluția normei gradientului și a pierderii
grad_norm_evol = zeros(1, epochs);
loss_evol = zeros(1, epochs);
time_evol = zeros(1, epochs); % Pentru urmărirea timpului
for epoch = 1:epochs
       start_time = tic;% Începerea cronometrarii pentru epocă
   % Amestecarea datelor și etichetelor de antrenament la începutul fiecărei epoci
    perm = randperm(size(X_train_aug, 1));
    X_train_aug = X_train_aug(perm, :);
    y_train = y_train(perm);
    
    for i = 1:batch_size:size(X_train_aug, 1)
        % Selectarea unui mini-batch
        idx = i:min(i+batch_size-1, size(X_train_aug, 1));
        X_batch = X_train_aug(idx, :);
        y_batch = y_train(idx);

        % Propagare înainte pentru mini-batch
        Z_hidden = X_batch * W;
        A_hidden = g(Z_hidden);
        Y_pred = A_hidden * x;

        % Calculul pierderii pentru mini-batch
        loss = cross_entropy_loss(y_batch, Y_pred);
         loss_evol(epoch) = loss;

       % Backpropagation pentru a calcula gradientii pentru mini-batch
        dY_pred = - (y_batch ./ Y_pred - (1 - y_batch) ./ (1 - Y_pred));
        dA_hidden = dY_pred * x' .* g_prime(Z_hidden); % Aplicarea derivatei funcției de activare g
        grad_x = A_hidden' * dY_pred / numel(y_batch);
        grad_W = X_batch' * (dA_hidden .* g_prime(Z_hidden)) / numel(y_batch);

        % Actualizarea parametriilor
        x = x - learning_rate * grad_x;
        W = W - learning_rate * grad_W;
    end
    grad_norm_evol(epoch) = norm(grad_W, 'fro'); % Norma Frobenius
    stop_time = toc(start_time);
    time_evol(epoch) = stop_time; 
    if mod(epoch, decay_steps) == 0
        learning_rate = learning_rate * decay_rate;
    end
end
% Calcul istoric cumulativ al timpului
cumulative_time_evol = cumsum(time_evol);
% Plotarea evoluției pierderii și a normei gradientului
figure;
subplot(2,1,1);
plot(1:epochs, loss_evol);
title('Evoluția Pierderii');
xlabel('Epoch');
ylabel('Pierdere');

subplot(2,1,2);
plot(1:epochs, grad_norm_evol);
title('Evoluția Normei Gradientului');
xlabel('Epoch');
ylabel('Norma Gradientului');
% Plotarea evoluției pierderii în timp
figure;
subplot(2,1,1);
plot(cumulative_time_evol, loss_evol);
title('Evoluția Pierderii în Timp');
xlabel('Timp (secunde)');
ylabel('Pierdere');

% Plotarea evoluției normei gradientului în timp
subplot(2,1,2);
plot(cumulative_time_evol, grad_norm_evol);
title('Evoluția Normei Gradientului în Timp');
xlabel('Timp (secunde)');
ylabel('Norma Gradientului');

sgtitle('Progresul Antrenării de-a Lungul Timpului'); 


% Normalizarea datelor de testare 
X_test_normalized = (X_test - mu) ./ sigma;
X_test_aug = [ones(size(X_test, 1), 1), X_test_normalized]; % Adaugă coloana de bias

% Calculul predicțiilor
Z_hidden_test = X_test_aug * W;
A_hidden_test = g(Z_hidden_test);
Y_pred_test = A_hidden_test * x; % Ieșirile modelului pentru setul de testare

% Convertirea ieșirilor modelului în clasificări binare
predictions = Y_pred_test > 0.5; % Prag de 0.5

% Convertirea predicțiile în același tip de date ca și etichetele de testare
predictions = double(predictions);

% y_test sa fie de același tip cu predictions
y_test = double(y_test);

% Calculul acurateței
accuracy = mean(predictions == y_test); % Procentul de corectitudine

% Afișarea acurateței
fprintf('Acuratețea pe setul de testare este: %.2f%%\n', accuracy * 100);

% Crearea matricei de confuzie
confusionMatrix = confusionmat(y_test, predictions);

% Afișarea matricei de confuzie
disp(confusionMatrix);

% Definirea adevărat pozitive (TP), fals pozitive (FP), adevărat negative (TN), și fals negative (FN)
TP = confusionMatrix(1,1);
FN = confusionMatrix(1,2);
FP = confusionMatrix(2,1);
TN = confusionMatrix(2,2);

% Calculul preciziei
precision = TP / (TP + FP);

% Calculul reamintirii
recall = TP / (TP + FN);

% Calculul scorului F1
F1_score = 2 * (precision * recall) / (precision + recall);

% Afișarea rezultatelor
fprintf('Precizia: %.2f\n', precision);
fprintf('Reamintirea: %.2f\n', recall);
fprintf('Scorul F1: %.2f\n', F1_score);


%% Metoda gradient descendent
clc;
clear;
close all;

% Încarcarea setului de date
filename = 'iris.data';
opts = detectImportOptions(filename, 'FileType', 'text', 'Delimiter', ',');
opts.VariableTypes(5) = {'categorical'}; 
data = readtable(filename, opts);

% Amestecarea rândurilor pentru a evita problemele de ordonare
data = data(randperm(size(data, 1)), :);

% Extragere caracteristici și etichete
features = table2array(data(:, 1:4));
species = data.Var5; % Accesarea coloanei speciilor folosind notația cu punct

% Convertire etichete categorice în binar: 1 pentru Iris-setosa și 0 pentru celelalte
binaryLabels = double(species == 'Iris-setosa');

% Împărțirea setului de date în set de antrenament și set de testare (80%-20%)
cv = cvpartition(size(features, 1), 'HoldOut', 0.2);
idx = cv.test;

% Set de antrenament
X_train = features(~idx, :);
y_train = binaryLabels(~idx);

% Set de testare
X_test = features(idx, :);
y_test = binaryLabels(idx);

% Normalizare date de antrenament și testare
[X_train, mu, sigma] = zscore(X_train);
X_test = (X_test - mu) ./ sigma;

% Inițializare parametri rețea
m = size(X_train, 2); % Număr de caracteristici
n = 10; % Număr de neuroni în stratul ascuns
W = randn(m + 1, n) * 0.01; % Ponderi incluzând termenul de bias
x = randn(n, 1) * 0.01; % Ponderi strat de ieșire
X_train_aug = [ones(size(X_train, 1), 1), X_train];  % Date de intrare 

% Funcția de activare Aranda-Ordaz
a_param = 1;
 g = @(z) 1 - (1 + a_param * exp(z)).^(1/a_param);
g_prime = @(z) - exp(z) .* (1 + a_param * exp(z)).^(1/a_param - 1);
% Funcția de pierdere entropie încrucișată binară
cross_entropy_loss = @(y, y_pred) -mean(y .* log(y_pred) + (1 - y) .* log(1 - y_pred));

initial_learning_rate = 0.01; % Rata inițială de învățare
decay_rate = 0.9;  % Factorul de descreștere a ratei de învățare
decay_steps = 1000; % Descreștere rată de învățare la fiecare 1000 de epoci
epochs = 10000;% Numărul de epoci
learning_rate = initial_learning_rate;% Rata de învățare curentă
grad_norm_evol = zeros(1, epochs);
loss_evol = zeros(1, epochs);
time_evol = zeros(1, epochs); % Pentru urmărirea timpului

for epoch = 1:epochs
    start_time = tic; % Începerea cronometrarii pentru epocă
    % Propagare înainte
    Z_hidden = X_train_aug * W;
    A_hidden = g(Z_hidden);
    Y_pred = A_hidden * x;

    % Calculul pierderii
    loss = cross_entropy_loss(y_train, Y_pred);
    loss_evol(epoch) = loss;
    dY_pred = - (y_train ./ Y_pred - (1 - y_train) ./ (1 - Y_pred));
    dA_hidden = dY_pred * x' .* g_prime(Z_hidden); % Aplicarea derivatei funcției de activare g
    grad_x = A_hidden' * dY_pred / numel(y_train);
    grad_W = X_train_aug' * (dA_hidden .* g_prime(Z_hidden)) / numel(y_train); 
    % Actualizarea parametriilor
    x = x - learning_rate * grad_x;
    W = W - learning_rate * grad_W;

    grad_norm_evol(epoch) = norm(grad_W, 'fro');% Norma Frobenius
    stop_time = toc(start_time);
    time_evol(epoch) = stop_time; 

    if mod(epoch, decay_steps) == 0
        learning_rate = learning_rate * decay_rate;
    end
end
cumulative_time_evol = cumsum(time_evol);

% Plotarea evoluției pierderii și a normei gradientului
figure;
subplot(2,1,1);
plot(1:epochs, loss_evol);
title('Evoluția Pierderii');
xlabel('Epoch');
ylabel('Pierdere');

subplot(2,1,2);
plot(1:epochs, grad_norm_evol);
title('Evoluția Normei Gradientului');
xlabel('Epoch');
ylabel('Norma Gradientului');
% Plotarea evoluției pierderii în timp
figure;
subplot(2,1,1);
plot(cumulative_time_evol, loss_evol);
title('Evoluția Pierderii în Timp');
xlabel('Timp (secunde)');
ylabel('Pierdere');

% Plotarea evoluției normei gradientului în timp
subplot(2,1,2);
plot(cumulative_time_evol, grad_norm_evol);
title('Evoluția Normei Gradientului în Timp');
xlabel('Timp (secunde)');
ylabel('Norma Gradientului');

sgtitle('Progresul Antrenării de-a Lungul Timpului');  

% Normalizarea datelor de testare 
X_test_normalized = (X_test - mu) ./ sigma;
X_test_aug = [ones(size(X_test, 1), 1), X_test_normalized]; % Adaugă coloana de bias

% Calculul predicțiilor
Z_hidden_test = X_test_aug * W;
A_hidden_test = g(Z_hidden_test);
Y_pred_test = A_hidden_test * x; % Ieșirile modelului pentru setul de testare

% Convertirea ieșirilor modelului în clasificări binare
predictions = Y_pred_test > 0.5; % Prag de 0.5

% Convertirea predicțiile în același tip de date ca și etichetele de testare
predictions = double(predictions);

% y_test sa fie de același tip cu predictions
y_test = double(y_test);

% Calculul acurateței
accuracy = mean(predictions == y_test); % Procentul de corectitudine

% Afișarea acurateței
fprintf('Acuratețea pe setul de testare este: %.2f%%\n', accuracy * 100);

% Crearea matricei de confuzie
confusionMatrix = confusionmat(y_test, predictions);

% Afișarea matricei de confuzie
disp(confusionMatrix);

% Definirea adevărat pozitive (TP), fals pozitive (FP), adevărat negative (TN), și fals negative (FN)
TP = confusionMatrix(1,1);
FN = confusionMatrix(1,2);
FP = confusionMatrix(2,1);
TN = confusionMatrix(2,2);

% Calculul preciziei
precision = TP / (TP + FP);

% Calculul reamintirii
recall = TP / (TP + FN);

% Calculul scorului F1
F1_score = 2 * (precision * recall) / (precision + recall);

% Afișarea rezultatelor
fprintf('Precizia: %.2f\n', precision);
fprintf('Reamintirea: %.2f\n', recall);
fprintf('Scorul F1: %.2f\n', F1_score);



