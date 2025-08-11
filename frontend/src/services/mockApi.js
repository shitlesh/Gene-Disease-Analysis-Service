/**
 * Mock API service simulating gene-disease analysis with streaming results
 * Provides realistic async behavior with progressive updates
 */

/**
 * Simulates streaming gene-disease analysis with progressive updates
 * @param {string} gene - Gene name to analyze
 * @param {string} disease - Disease name to analyze
 * @param {string} apiKey - API key (validated but not used in mock)
 * @param {Function} onProgress - Callback for streaming progress updates
 * @returns {Promise} Final analysis result
 */
export const mockAnalyzeGeneDisease = async (gene, disease, apiKey, onProgress) => {
  // Validate inputs
  if (!gene || !disease || !apiKey) {
    throw new Error('Gene name, disease name, and API key are required');
  }
  
  // Simulate streaming analysis with realistic delays
  const progressSteps = [
    'Initializing analysis pipeline...',
    `Searching databases for gene: ${gene}...`,
    'Retrieving gene sequence and annotations...',
    `Analyzing disease associations for: ${disease}...`,
    'Cross-referencing genetic variants...',
    'Examining protein interactions...',
    'Analyzing pathway involvement...',
    'Evaluating therapeutic targets...',
    'Generating comprehensive report...',
    'Finalizing results...',
  ];
  
  // Stream progress updates with realistic timing
  for (let i = 0; i < progressSteps.length; i++) {
    await new Promise(resolve => setTimeout(resolve, 800 + Math.random() * 400));
    onProgress(progressSteps[i]);
  }
  
  // Generate mock comprehensive result
  const mockResult = generateMockAnalysisResult(gene, disease);
  
  // Final delay before completion
  await new Promise(resolve => setTimeout(resolve, 500));
  
  return {
    gene,
    disease,
    fullResult: mockResult,
    completedAt: new Date().toISOString(),
  };
};

/**
 * Generates realistic mock analysis results
 * @param {string} gene - Gene name
 * @param {string} disease - Disease name  
 * @returns {string} Mock analysis result
 */
const generateMockAnalysisResult = (gene, disease) => {
  const mockFindings = [
    `Gene ${gene} shows significant association with ${disease} development`,
    `Identified 3 pathogenic variants in ${gene} linked to ${disease} pathogenesis`,
    `${gene} protein expression is altered in 67% of ${disease} cases`,
    `Key pathway: ${gene} regulates inflammatory response in ${disease}`,
    `Potential therapeutic target: ${gene} inhibitors show promise`,
    `Clinical relevance: ${gene} mutations found in 23% of familial ${disease} cases`,
  ];
  
  // Select random findings for variety
  const selectedFindings = mockFindings
    .sort(() => 0.5 - Math.random())
    .slice(0, 3 + Math.floor(Math.random() * 3));
  
  return `Analysis Summary for ${gene} and ${disease}:\n\n${selectedFindings.map((finding, i) => `${i + 1}. ${finding}`).join('\n')}\n\nConfidence Score: ${(85 + Math.random() * 10).toFixed(1)}%\nLast Updated: ${new Date().toLocaleDateString()}`;
};