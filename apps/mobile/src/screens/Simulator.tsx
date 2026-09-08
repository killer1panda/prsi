import React, { useState } from 'react';
import { StyleSheet, Text, View, SafeAreaView, TextInput, TouchableOpacity, ActivityIndicator, ScrollView } from 'react-native';
import { simulateAttack } from '../api/client';

export default function Simulator() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleSimulate = async () => {
    if (!text) return;
    setLoading(true);
    try {
      const res = await simulateAttack(text);
      setResult(res);
    } catch (error) {
      setResult({ error: 'Failed to simulate attack' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Attack Simulator</Text>
          <Text style={styles.headerSubtitle}>Simulate threat scenarios</Text>
        </View>

        <View style={styles.inputCard}>
          <Text style={styles.label}>Scenario Description</Text>
          <TextInput
            style={styles.input}
            multiline
            placeholder="Describe the attack scenario..."
            placeholderTextColor="#64748B"
            value={text}
            onChangeText={setText}
          />
          
          <TouchableOpacity style={styles.simulateBtn} onPress={handleSimulate} disabled={loading || !text}>
            {loading ? <ActivityIndicator color="#FFF" /> : <Text style={styles.simulateBtnText}>Simulate Attack</Text>}
          </TouchableOpacity>
        </View>

        {result && (
          <View style={styles.resultCard}>
            <Text style={styles.resultTitle}>Simulation Output</Text>
            <Text style={styles.resultText}>{JSON.stringify(result, null, 2)}</Text>
          </View>
        )}
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
  inputCard: { backgroundColor: '#1E293B', borderRadius: 16, padding: 16, marginBottom: 24, borderWidth: 1, borderColor: '#334155' },
  label: { fontSize: 14, color: '#94A3B8', marginBottom: 8, fontWeight: '600' },
  input: { backgroundColor: '#0F172A', borderRadius: 8, padding: 12, color: '#F8FAFC', minHeight: 120, textAlignVertical: 'top', borderWidth: 1, borderColor: '#334155', marginBottom: 16 },
  simulateBtn: { backgroundColor: '#EF4444', borderRadius: 8, padding: 16, alignItems: 'center' },
  simulateBtnText: { color: '#FFF', fontSize: 16, fontWeight: 'bold' },
  resultCard: { backgroundColor: '#1E293B', borderRadius: 16, padding: 16, borderWidth: 1, borderColor: '#334155' },
  resultTitle: { fontSize: 16, fontWeight: 'bold', color: '#F8FAFC', marginBottom: 12 },
  resultText: { color: '#CBD5E1', fontFamily: 'monospace' },
});
