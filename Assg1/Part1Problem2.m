% Initialize environment
clear all;
close all;
clc;

% Configuration
outlierThreshold = 2; % Standard deviation threshold for outliers
fitTypes = {'linear', 'quadratic'};
numFits = length(fitTypes);

%% Section 1: Heating Oil Analysis

% Input Data
oilConsumption = [270 362 162 45 91 233 372 305 234 122 25 210 450 325 52];
temperature = [40 27 40 73 65 65 10 9 24 65 66 41 22 40 60];
insulationLevel = [4 4 10 6 7 40 6 10 10 4 10 6 4 4 10];

% Visualization: Scatter Plots
figure;
scatter(temperature, oilConsumption, 'filled');
xlabel('Temperature (°F)', 'FontSize', 14);
ylabel('Oil Consumption (gal)', 'FontSize', 14);
title('Oil Consumption vs Temperature', 'FontSize', 16);
grid on;

figure;
scatter(insulationLevel, oilConsumption, 'filled');
xlabel('Insulation Thickness (in)', 'FontSize', 14);
ylabel('Oil Consumption (gal)', 'FontSize', 14);
title('Oil Consumption vs Insulation', 'FontSize', 16);
grid on;

figure;
scatter3(temperature, insulationLevel, oilConsumption, 'filled');
xlabel('Temperature (°F)', 'FontSize', 14);
ylabel('Insulation Thickness (in)', 'FontSize', 14);
zlabel('Oil Consumption (gal)', 'FontSize', 14);
title('3D View: Oil Consumption vs Temp & Insulation', 'FontSize', 16);
grid on;

% Model Fitting
modelFits = cell(numFits, 1);
r2Values = zeros(numFits, 1);
modelCoeffs = cell(numFits, 1);

for idx = 1:numFits
    modelFits{idx} = fitlm([temperature', insulationLevel'], oilConsumption', fitTypes{idx});
    r2Values(idx) = modelFits{idx}.Rsquared.Ordinary;
    modelCoeffs{idx} = modelFits{idx}.Coefficients.Estimate;
end

% Display Model Fit Quality
disp('=== Model Fit Assessment ===');
for idx = 1:numFits
    fprintf('  %-10s Model | R²: %.4f\n', fitTypes{idx}, r2Values(idx));
end

% Identify Best Model
[~, topModelIdx] = max(r2Values);
fprintf('\n  Best Performing Model: %s\n', fitTypes{topModelIdx});

% Model Equations
disp('=== Model Equations ===');
for idx = 1:numFits
    coeffSet = modelCoeffs{idx};
    eqStr = sprintf('  %-10s | y = %.2f', fitTypes{idx}, coeffSet(1));
    for term = 2:length(coeffSet)
        if coeffSet(term) >= 0
            eqStr = [eqStr sprintf(' + %.2f*x%d', coeffSet(term), term-1)];
        else
            eqStr = [eqStr sprintf(' - %.2f*x%d', abs(coeffSet(term)), term-1)];
        end
    end
    disp(eqStr);
end

% Prediction Scenario
tempTest = 15; % °F
insulTest = 5; % inches
predictions = zeros(numFits, 1);

disp('=== Predictions ===');
for idx = 1:numFits
    predictions(idx) = predict(modelFits{idx}, [tempTest, insulTest]);
    fprintf('  %-10s Model | Temp: %d°F, Insulation: %d in | Oil: %.2f gal\n', ...
        fitTypes{idx}, tempTest, insulTest, predictions(idx));
end

% Manual Equation-Based Prediction
linCoeff = modelCoeffs{1};
predLinEq = linCoeff(1) + linCoeff(2)*tempTest + linCoeff(3)*insulTest;

quadCoeff = modelCoeffs{2};
predQuadEq = quadCoeff(1) + quadCoeff(2)*tempTest + quadCoeff(3)*insulTest + ...
    quadCoeff(4)*tempTest*insulTest + quadCoeff(5)*tempTest^2 + quadCoeff(6)*insulTest^2;

fprintf('\n  Equation-Based:\n');
fprintf('  %-10s | Temp: %d°F, Insulation: %d in | Oil: %.2f gal\n', ...
    fitTypes{1}, tempTest, insulTest, predLinEq);
fprintf('  %-10s | Temp: %d°F, Insulation: %d in | Oil: %.2f gal\n', ...
    fitTypes{2}, tempTest, insulTest, predQuadEq);



% Visualization with Best Fit
figure;
scatter3(temperature, insulationLevel, oilConsumption, 'filled');
hold on;
[X, Y] = meshgrid(min(temperature):1:max(temperature), min(insulationLevel):1:max(insulationLevel));
Z = predict(modelFits{topModelIdx}, [X(:), Y(:)]);
Z = reshape(Z, size(X));
mesh(X, Y, Z, 'FaceAlpha', 0.5);
xlabel('Temperature (°F)', 'FontSize', 14);
ylabel('Insulation Thickness (in)', 'FontSize', 14);
zlabel('Oil Consumption (gal)', 'FontSize', 14);
title('Best Fit Model Surface', 'FontSize', 16);
legend('Data', 'Fit Surface', 'Location', 'northwest');
grid on;
hold off;

%% Section 2: Outlier Removal and Re-Analysis

% Outlier Detection
tempMean = mean(temperature);
tempStd = std(temperature);
insulMean = mean(insulationLevel);
insulStd = std(insulationLevel);

tempOutliers = (temperature < (tempMean - outlierThreshold*tempStd)) | ...
    (temperature > (tempMean + outlierThreshold*tempStd));
insulOutliers = (insulationLevel < (insulMean - outlierThreshold*insulStd)) | ...
    (insulationLevel > (insulMean + outlierThreshold*insulStd));
outlierMask = tempOutliers | insulOutliers;

% Cleaned Dataset
oilClean = oilConsumption(~outlierMask);
tempClean = temperature(~outlierMask);
insulClean = insulationLevel(~outlierMask);

% Visualization: Cleaned Data
figure;
scatter(tempClean, oilClean, 'filled');
xlabel('Temperature (°F)', 'FontSize', 14);
ylabel('Oil Consumption (gal)', 'FontSize', 14);
title('Cleaned: Oil vs Temperature', 'FontSize', 16);
grid on;

figure;
scatter(insulClean, oilClean, 'filled');
xlabel('Insulation Thickness (in)', 'FontSize', 14);
ylabel('Oil Consumption (gal)', 'FontSize', 14);
title('Cleaned: Oil vs Insulation', 'FontSize', 16);
grid on;

figure;
scatter3(tempClean, insulClean, oilClean, 'filled');
xlabel('Temperature (°F)', 'FontSize', 14);
ylabel('Insulation Thickness (in)', 'FontSize', 14);
zlabel('Oil Consumption (gal)', 'FontSize', 14);
title('Cleaned 3D View: Oil vs Temp & Insulation', 'FontSize', 16);
grid on;

% Model Fitting on Cleaned Data
cleanFits = cell(numFits, 1);
r2Clean = zeros(numFits, 1);
coeffClean = cell(numFits, 1);

for idx = 1:numFits
    cleanFits{idx} = fitlm([tempClean', insulClean'], oilClean', fitTypes{idx});
    r2Clean(idx) = cleanFits{idx}.Rsquared.Ordinary;
    coeffClean{idx} = cleanFits{idx}.Coefficients.Estimate;
end

% Display Cleaned Model Fit Quality
disp('=== Cleaned Data Model Fit ===');
for idx = 1:numFits
    fprintf('  %-10s Model | R²: %.4f\n', fitTypes{idx}, r2Clean(idx));
end

% Identify Best Cleaned Model
[~, topCleanIdx] = max(r2Clean);
fprintf('\n  Best Performing Model (Cleaned): %s\n', fitTypes{topCleanIdx});

% Cleaned Model Equations
disp('=== Cleaned Model Equations ===');
for idx = 1:numFits
    coeffSet = coeffClean{idx};
    eqStr = sprintf('  %-10s | y = %.2f', fitTypes{idx}, coeffSet(1));
    for term = 2:length(coeffSet)
        if coeffSet(term) >= 0
            eqStr = [eqStr sprintf(' + %.2f*x%d', coeffSet(term), term-1)];
        else
            eqStr = [eqStr sprintf(' - %.2f*x%d', abs(coeffSet(term)), term-1)];
        end
    end
    disp(eqStr);
end

% Prediction on Cleaned Data
cleanPreds = zeros(numFits, 1);
disp('=== Cleaned Data Predictions ===');
for idx = 1:numFits
    cleanPreds(idx) = predict(cleanFits{idx}, [tempTest, insulTest]);
    fprintf('  %-10s Model | Temp: %d°F, Insulation: %d in | Oil: %.2f gal\n', ...
        fitTypes{idx}, tempTest, insulTest, cleanPreds(idx));
end

% Manual Equation-Based Prediction (Cleaned)
linCoeffClean = coeffClean{1};
predLinEqClean = linCoeffClean(1) + linCoeffClean(2)*tempTest + linCoeffClean(3)*insulTest;

quadCoeffClean = coeffClean{2};
predQuadEqClean = quadCoeffClean(1) + quadCoeffClean(2)*tempTest + quadCoeffClean(3)*insulTest + ...
    quadCoeffClean(4)*tempTest*insulTest + quadCoeffClean(5)*tempTest^2 + quadCoeffClean(6)*insulTest^2;

fprintf('\n  Equation-Based (Cleaned):\n');
fprintf('  %-10s | Temp: %d°F, Insulation: %d in | Oil: %.2f gal\n', ...
    fitTypes{1}, tempTest, insulTest, predLinEqClean);
fprintf('  %-10s | Temp: %d°F, Insulation: %d in | Oil: %.2f gal\n', ...
    fitTypes{2}, tempTest, insulTest, predQuadEqClean);



% Visualization with Best Cleaned Fit
figure;
scatter3(tempClean, insulClean, oilClean, 'filled');
hold on;
[Xc, Yc] = meshgrid(min(tempClean):1:max(tempClean), min(insulClean):1:max(insulClean));
Zc = predict(cleanFits{topCleanIdx}, [Xc(:), Yc(:)]);
Zc = reshape(Zc, size(Xc));
mesh(Xc, Yc, Zc, 'FaceAlpha', 0.5);
xlabel('Temperature (°F)', 'FontSize', 14);
ylabel('Insulation Thickness (in)', 'FontSize', 14);
zlabel('Oil Consumption (gal)', 'FontSize', 14);
title('Cleaned Data with Best Fit Surface', 'FontSize', 16);
legend('Cleaned Data', 'Fit Surface', 'Location', 'northwest');
grid on;
hold off;