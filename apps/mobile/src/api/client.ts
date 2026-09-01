export const fetchRiskData = async () => {
  // Dummy implementation for fetching risk data
  return [
    { id: 1, title: 'Cyber Security', score: 92, status: 'Critical', color: '#EF4444' },
    { id: 2, title: 'Market Volatility', score: 78, status: 'High', color: '#F97316' },
    { id: 3, title: 'Operational', score: 45, status: 'Moderate', color: '#EAB308' },
    { id: 4, title: 'Compliance', score: 20, status: 'Low', color: '#22C55E' },
  ];
};

export const fetchOverallRisk = async () => {
  return {
    score: 84,
    status: 'Elevated Threat Level'
  };
};
