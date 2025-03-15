%% Initialization and Setup
close all;  % Close all figures
clear;      % Clear workspace variables
clc;        % Clear command window

%% Configuration Constants
OUTLIER_THRESHOLD = 2;          % Standard deviations for outlier detection
MODEL_TYPES = {'linear', 'quadratic', 'cubic'};  % Regression model types
NUM_MODELS = length(MODEL_TYPES);  % Number of models to evaluate
R2_GAIN_THRESHOLD = 0.05;       % Threshold for R² gain to justify complexity

%% Data Preparation for Insurance Analysis
yearRange = 1987:1996;          % Full year range
insuredData = [13300, 12400, 10900, 10100, 10150, 10000, 800, 9000, 8750, 8100];
relativeYears = yearRange - 1987;  % Normalize years relative to 1987

%% Visualization 1: Initial Data Scatter Plot
fig1 = figure('Position', [100, 100, 600, 450]);
scatter(relativeYears, insuredData, 60, 'filled', 'MarkerFaceColor', [0.3 0.6 0.9]);
hold on;
dataMean = mean(insuredData);
dataStd = std(insuredData);
isOutlier = (insuredData < (dataMean - OUTLIER_THRESHOLD * dataStd)) | ...
            (insuredData > (dataMean + OUTLIER_THRESHOLD * dataStd));
for idx = 1:length(yearRange)
    text(relativeYears(idx) + 0.15, insuredData(idx), num2str(yearRange(idx)), 'FontSize', 9);
    if isOutlier(idx)
        plot(relativeYears(idx), insuredData(idx), 'ro', 'MarkerSize', 12, 'LineWidth', 1.5);
    end
end
xlabel('Years Since 1987', 'FontSize', 11);
ylabel('Insured Individuals', 'FontSize', 11);
title('Insured Persons Trend (1987-1996)', 'FontSize', 12);
grid on;
box on;
set(gca, 'FontSize', 10);
hold off;

%% Model Fitting and Evaluation
modelFits = cell(NUM_MODELS, 1);
rSquaredValues = zeros(NUM_MODELS, 1);
adjustedRSquaredValues = zeros(NUM_MODELS, 1);
modelCoefficients = cell(NUM_MODELS, 1);
warning('off', 'stats:fitlm:RankDeficient');
warning('off', 'stats:fitlm:IllConditioned');
for modelIdx = 1:NUM_MODELS
    if strcmp(MODEL_TYPES{modelIdx}, 'cubic')
        [fitResult, goodness] = fit(yearRange', insuredData', 'poly3');
        rSquaredValues(modelIdx) = goodness.rsquare;
        n = length(yearRange);
        p = 4;
        adjustedRSquaredValues(modelIdx) = 1 - (1 - goodness.rsquare) * (n - 1) / (n - p - 1);
        modelCoefficients{modelIdx} = [fitResult.p4; fitResult.p3; fitResult.p2; fitResult.p1];
    else
        modelFits{modelIdx} = fitlm(yearRange, insuredData, MODEL_TYPES{modelIdx});
        rSquaredValues(modelIdx) = modelFits{modelIdx}.Rsquared.Ordinary;
        adjustedRSquaredValues(modelIdx) = modelFits{modelIdx}.Rsquared.Adjusted;
        modelCoefficients{modelIdx} = modelFits{modelIdx}.Coefficients.Estimate;
    end
end

% Display analysis results
fprintf('\n=== Initial Data Model Analysis ===\n');
fprintf('R-squared and Adjusted R-squared Results:\n');
for modelIdx = 1:NUM_MODELS
    fprintf('  %-10s Model: R² = %.4f, Adjusted R² = %.4f\n', ...
            MODEL_TYPES{modelIdx}, rSquaredValues(modelIdx), adjustedRSquaredValues(modelIdx));
end

% Custom model selection based on R² gain
if (rSquaredValues(3) - rSquaredValues(2)) < R2_GAIN_THRESHOLD
    bestModelIdx = 2; % Quadratic
    fprintf('\nR² gain from quadratic to cubic (%.4f) < threshold (%.2f), choosing simpler model.\n', ...
            rSquaredValues(3) - rSquaredValues(2), R2_GAIN_THRESHOLD);
else
    bestModelIdx = 3; % Cubic
    fprintf('\nR² gain from quadratic to cubic (%.4f) >= threshold (%.2f), choosing cubic.\n', ...
            rSquaredValues(3) - rSquaredValues(2), R2_GAIN_THRESHOLD);
end
fprintf('Optimal Model: %s\n', MODEL_TYPES{bestModelIdx});

% Print model equations
fprintf('\nModel Equations:\n');
for modelIdx = 1:NUM_MODELS
    coeffs = modelCoefficients{modelIdx};
    fprintf('  %-10s Model: y = ', MODEL_TYPES{modelIdx});
    for term = 1:length(coeffs)
        if term == 1
            fprintf('%.4f', coeffs(term));
        else
            fprintf(' %+.4f*x^%d', coeffs(term), term-1);
        end
    end
    fprintf('\n');
end

%% Visualization 2: Best Fit Model with Data
fig2 = figure('Position', [100, 100, 600, 450]);
smoothX = linspace(min(relativeYears), max(relativeYears), 100);
if strcmp(MODEL_TYPES{bestModelIdx}, 'cubic')
    smoothY = feval(fitResult, smoothX + 1987);
    plot(smoothX, smoothY, 'Color', [0.85, 0.33, 0.10], 'LineWidth', 2);
else
    smoothY = polyval(flipud(modelCoefficients{bestModelIdx}), smoothX + 1987);
    plot(smoothX, smoothY, 'Color', [0.85, 0.33, 0.10], 'LineWidth', 2);
end
hold on;
scatter(relativeYears, insuredData, 60, 'filled', 'MarkerFaceColor', [0.3 0.6 0.9]);
for idx = 1:length(yearRange)
    text(relativeYears(idx) + 0.15, insuredData(idx), num2str(yearRange(idx)), 'FontSize', 9);
    if isOutlier(idx)
        plot(relativeYears(idx), insuredData(idx), 'ro', 'MarkerSize', 12, 'LineWidth', 1.5);
    end
end
xlabel('Years Since 1987', 'FontSize', 11);
ylabel('Insured Individuals', 'FontSize', 11);
title(['Trend with ' MODEL_TYPES{bestModelIdx} ' Fit'], 'FontSize', 12);
legend('Model Fit', 'Data', 'Location', 'best');
grid on;
box on;
set(gca, 'FontSize', 10);
hold off;

%% Prediction for 1997
prediction1997 = polyval(flipud(modelCoefficients{bestModelIdx}), 1997);
fprintf('\nPredicted Insured Persons for 1997: %.2f\n', prediction1997);

%% Analysis Without Outliers
cleanYears = yearRange(~isOutlier);
cleanRelativeYears = relativeYears(~isOutlier);
cleanInsuredData = insuredData(~isOutlier);

%% Visualization 3: Cleaned Data Scatter Plot
fig3 = figure('Position', [100, 100, 600, 450]);
scatter(cleanRelativeYears, cleanInsuredData, 60, 'filled', 'MarkerFaceColor', [0.3 0.6 0.9]);
hold on;
for idx = 1:length(cleanYears)
    text(cleanRelativeYears(idx) + 0.15, cleanInsuredData(idx), num2str(cleanYears(idx)), 'FontSize', 9);
end
xlabel('Years Since 1987', 'FontSize', 11);
ylabel('Insured Individuals', 'FontSize', 11);
title('Cleaned Data (Outliers Removed)', 'FontSize', 12);
ylim([7500 14000]);
grid on;
box on;
set(gca, 'FontSize', 10);
hold off;

%% Cleaned Data Model Fitting
cleanModelFits = cell(NUM_MODELS, 1);
cleanRSquared = zeros(NUM_MODELS, 1);
cleanAdjustedRSquared = zeros(NUM_MODELS, 1);
cleanCoefficients = cell(NUM_MODELS, 1);
for modelIdx = 1:NUM_MODELS
    if strcmp(MODEL_TYPES{modelIdx}, 'cubic')
        [cleanFitResult, cleanGoodness] = fit(cleanYears', cleanInsuredData', 'poly3');
        cleanRSquared(modelIdx) = cleanGoodness.rsquare;
        n = length(cleanYears);
        p = 4;
        cleanAdjustedRSquared(modelIdx) = 1 - (1 - cleanGoodness.rsquare) * (n - 1) / (n - p - 1);
        cleanCoefficients{modelIdx} = [cleanFitResult.p4; cleanFitResult.p3; cleanFitResult.p2; cleanFitResult.p1];
    else
        cleanModelFits{modelIdx} = fitlm(cleanYears, cleanInsuredData, MODEL_TYPES{modelIdx});
        cleanRSquared(modelIdx) = cleanModelFits{modelIdx}.Rsquared.Ordinary;
        cleanAdjustedRSquared(modelIdx) = cleanModelFits{modelIdx}.Rsquared.Adjusted;
        cleanCoefficients{modelIdx} = cleanModelFits{modelIdx}.Coefficients.Estimate;
    end
end

% Display cleaned data results
fprintf('\n=== Cleaned Data Model Analysis ===\n');
fprintf('R-squared and Adjusted R-squared Results:\n');
for modelIdx = 1:NUM_MODELS
    fprintf('  %-10s Model: R² = %.4f, Adjusted R² = %.4f\n', ...
            MODEL_TYPES{modelIdx}, cleanRSquared(modelIdx), cleanAdjustedRSquared(modelIdx));
end

% Custom model selection based on R² gain
if (cleanRSquared(3) - cleanRSquared(2)) < R2_GAIN_THRESHOLD
    bestCleanModelIdx = 2; % Quadratic
    fprintf('\nR² gain from quadratic to cubic (%.4f) < threshold (%.2f), choosing simpler model.\n', ...
            cleanRSquared(3) - cleanRSquared(2), R2_GAIN_THRESHOLD);
else
    bestCleanModelIdx = 3; % Cubic
    fprintf('\nR² gain from quadratic to cubic (%.4f) >= threshold (%.2f), choosing cubic.\n', ...
            cleanRSquared(3) - cleanRSquared(2), R2_GAIN_THRESHOLD);
end
fprintf('Optimal Model (Cleaned): %s\n', MODEL_TYPES{bestCleanModelIdx});

% Print cleaned model equations
fprintf('\nCleaned Model Equations:\n');
for modelIdx = 1:NUM_MODELS
    coeffs = cleanCoefficients{modelIdx};
    fprintf('  %-10s Model: y = ', MODEL_TYPES{modelIdx});
    for term = 1:length(coeffs)
        if term == 1
            fprintf('%.4f', coeffs(term));
        else
            fprintf(' %+.4f*x^%d', coeffs(term), term-1);
        end
    end
    fprintf('\n');
end

%% Visualization 4: Cleaned Data with Best Fit
fig4 = figure('Position', [100, 100, 600, 450]);
cleanSmoothX = linspace(min(cleanRelativeYears), max(cleanRelativeYears), 100);
if strcmp(MODEL_TYPES{bestCleanModelIdx}, 'cubic')
    cleanSmoothY = feval(cleanFitResult, cleanSmoothX + 1987);
    plot(cleanSmoothX, cleanSmoothY, 'Color', [0.85, 0.33, 0.10], 'LineWidth', 2);
else
    cleanSmoothY = polyval(flipud(cleanCoefficients{bestCleanModelIdx}), cleanSmoothX + 1987);
    plot(cleanSmoothX, cleanSmoothY, 'Color', [0.85, 0.33, 0.10], 'LineWidth', 2);
end
hold on;
scatter(cleanRelativeYears, cleanInsuredData, 60, 'filled', 'MarkerFaceColor', [0.3 0.6 0.9]);
for idx = 1:length(cleanYears)
    text(cleanRelativeYears(idx) + 0.15, cleanInsuredData(idx), num2str(cleanYears(idx)), 'FontSize', 9);
end
xlabel('Years Since 1987', 'FontSize', 11);
ylabel('Insured Individuals', 'FontSize', 11);
title(['Cleaned Data with ' MODEL_TYPES{bestCleanModelIdx} ' Fit'], 'FontSize', 12);
legend('Model Fit', 'Data', 'Location', 'best');
ylim([7500 14000]);
grid on;
box on;
set(gca, 'FontSize', 10);
hold off;

%% Prediction for 1997 with Cleaned Data
cleanPrediction1997 = polyval(flipud(cleanCoefficients{bestCleanModelIdx}), 1997);
fprintf('\nPredicted Insured Persons for 1997 (Cleaned): %.2f\n', cleanPrediction1997);

warning('on', 'stats:fitlm:RankDeficient');
warning('on', 'stats:fitlm:IllConditioned');