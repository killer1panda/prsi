const BASE_URL = 'http://localhost:8000';
const HEADERS = {
  'Content-Type': 'application/json',
  'Authorization': 'Bearer dummy_token',
};

export const fetchRiskData = async () => {
  try {
    const response = await fetch(`${BASE_URL}/dashboard/leaderboard`, {
      method: 'GET',
      headers: HEADERS,
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    const data = await response.json();
    return data.leaderboard || [];
  } catch (error) {
    console.error("fetchRiskData error:", error);
    // Fallback data in case server is not running during testing
    return [
      { id: 1, title: 'Cyber Security', score: 92, status: 'Critical', color: '#EF4444' },
      { id: 2, title: 'Market Volatility', score: 78, status: 'High', color: '#F97316' },
      { id: 3, title: 'Operational', score: 45, status: 'Moderate', color: '#EAB308' },
      { id: 4, title: 'Compliance', score: 20, status: 'Low', color: '#22C55E' },
    ];
  }
};

export const fetchOverallRisk = async () => {
  return {
    score: 84,
    status: 'Elevated Threat Level'
  };
};

export const analyzeThreat = async (text: string, source: string = 'reddit') => {
  try {
    const response = await fetch(`${BASE_URL}/analyze`, {
      method: 'POST',
      headers: HEADERS,
      body: JSON.stringify({ text, source }),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("analyzeThreat error:", error);
    throw error;
  }
};

export const simulateAttack = async (text: string) => {
  try {
    const response = await fetch(`${BASE_URL}/attack/simulate`, {
      method: 'POST',
      headers: HEADERS,
      body: JSON.stringify({ text }),
    });
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error("simulateAttack error:", error);
    throw error;
  }
};
