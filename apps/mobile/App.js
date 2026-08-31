import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { StyleSheet, Text, View, ScrollView, SafeAreaView, TouchableOpacity } from 'react-native';

export default function App() {
  const riskFactors = [
    { id: 1, title: 'Cyber Security', score: 92, status: 'Critical', color: '#EF4444' },
    { id: 2, title: 'Market Volatility', score: 78, status: 'High', color: '#F97316' },
    { id: 3, title: 'Operational', score: 45, status: 'Moderate', color: '#EAB308' },
    { id: 4, title: 'Compliance', score: 20, status: 'Low', color: '#22C55E' },
  ];

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="light" />
      <ScrollView contentContainerStyle={styles.container}>
        
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Risk Dashboard</Text>
          <Text style={styles.headerSubtitle}>Real-time global threat assessment</Text>
        </View>

        <View style={styles.mainCard}>
          <Text style={styles.mainCardTitle}>Overall Risk Index</Text>
          <View style={styles.scoreContainer}>
            <Text style={styles.scoreText}>84</Text>
            <Text style={styles.scoreMax}>/100</Text>
          </View>
          <Text style={styles.mainCardStatus}>Elevated Threat Level</Text>
        </View>

        <Text style={styles.sectionTitle}>Key Risk Factors</Text>
        
        {riskFactors.map((item) => (
          <TouchableOpacity key={item.id} style={styles.riskCard} activeOpacity={0.8}>
            <View style={styles.riskCardHeader}>
              <Text style={styles.riskCardTitle}>{item.title}</Text>
              <View style={[styles.badge, { backgroundColor: item.color + '20' }]}>
                <Text style={[styles.badgeText, { color: item.color }]}>{item.status}</Text>
              </View>
            </View>
            
            <View style={styles.progressBarContainer}>
              <View style={[styles.progressBar, { width: `${item.score}%`, backgroundColor: item.color }]} />
            </View>
            <View style={styles.scoreRow}>
              <Text style={styles.scoreLabel}>Risk Score</Text>
              <Text style={styles.scoreValue}>{item.score}</Text>
            </View>
          </TouchableOpacity>
        ))}

      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#0F172A' },
  container: { padding: 20 },
  header: { marginBottom: 24, marginTop: 12 },
  headerTitle: { fontSize: 28, fontWeight: 'bold', color: '#F8FAFC', marginBottom: 4 },
  headerSubtitle: { fontSize: 15, color: '#94A3B8' },
  mainCard: { backgroundColor: '#1E293B', borderRadius: 20, padding: 24, alignItems: 'center', marginBottom: 32, borderWidth: 1, borderColor: '#334155' },
  mainCardTitle: { fontSize: 16, color: '#CBD5E1', fontWeight: '600', marginBottom: 12 },
  scoreContainer: { flexDirection: 'row', alignItems: 'baseline' },
  scoreText: { fontSize: 64, fontWeight: '800', color: '#F8FAFC' },
  scoreMax: { fontSize: 24, color: '#64748B', fontWeight: '600', marginLeft: 4 },
  mainCardStatus: { marginTop: 8, fontSize: 16, color: '#EF4444', fontWeight: '700' },
  sectionTitle: { fontSize: 20, fontWeight: 'bold', color: '#F8FAFC', marginBottom: 16 },
  riskCard: { backgroundColor: '#1E293B', borderRadius: 16, padding: 16, marginBottom: 16, borderWidth: 1, borderColor: '#334155' },
  riskCardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 },
  riskCardTitle: { fontSize: 17, fontWeight: '600', color: '#F8FAFC' },
  badge: { paddingHorizontal: 10, paddingVertical: 4, borderRadius: 12 },
  badgeText: { fontSize: 12, fontWeight: 'bold' },
  progressBarContainer: { height: 8, backgroundColor: '#334155', borderRadius: 4, overflow: 'hidden', marginBottom: 12 },
  progressBar: { height: '100%', borderRadius: 4 },
  scoreRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  scoreLabel: { fontSize: 14, color: '#94A3B8' },
  scoreValue: { fontSize: 16, fontWeight: 'bold', color: '#F8FAFC' }
});
