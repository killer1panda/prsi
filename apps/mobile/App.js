import React, { useState } from 'react';
import { StyleSheet, View, TouchableOpacity, Text, SafeAreaView } from 'react-native';
import Dashboard from './src/screens/Dashboard';
import Analyzer from './src/screens/Analyzer';
import Simulator from './src/screens/Simulator';

export default function App() {
  const [currentTab, setCurrentTab] = useState('dashboard');

  const renderScreen = () => {
    switch (currentTab) {
      case 'dashboard':
        return <Dashboard />;
      case 'analyzer':
        return <Analyzer />;
      case 'simulator':
        return <Simulator />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.screenContainer}>
        {renderScreen()}
      </View>
      <View style={styles.tabBar}>
        <TouchableOpacity 
          style={styles.tab} 
          onPress={() => setCurrentTab('dashboard')}
        >
          <Text style={[styles.tabText, currentTab === 'dashboard' && styles.tabTextActive]}>Dashboard</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={styles.tab} 
          onPress={() => setCurrentTab('analyzer')}
        >
          <Text style={[styles.tabText, currentTab === 'analyzer' && styles.tabTextActive]}>Analyzer</Text>
        </TouchableOpacity>
        
        <TouchableOpacity 
          style={styles.tab} 
          onPress={() => setCurrentTab('simulator')}
        >
          <Text style={[styles.tabText, currentTab === 'simulator' && styles.tabTextActive]}>Simulator</Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0F172A',
  },
  screenContainer: {
    flex: 1,
  },
  tabBar: {
    flexDirection: 'row',
    backgroundColor: '#1E293B',
    borderTopWidth: 1,
    borderTopColor: '#334155',
    paddingVertical: 12,
    paddingBottom: 24, // Safe area padding for bottom
  },
  tab: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  tabText: {
    color: '#64748B',
    fontSize: 14,
    fontWeight: '600',
  },
  tabTextActive: {
    color: '#3B82F6',
  }
});
