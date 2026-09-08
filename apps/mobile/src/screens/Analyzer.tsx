import React, { useState } from 'react';
import { StyleSheet, Text, View, SafeAreaView, TextInput, TouchableOpacity, ActivityIndicator, ScrollView } from 'react-native';
import { analyzeThreat } from '../api/client';

export default function Analyzer() {
  const [text, setText] = useState('');
  const [source, setSource] = useState('reddit');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const handleAnalyze = async () => {
    if (!text) return;
    setLoading(true);
    try {
      const res = await analyzeThreat(text, source);
      setResult(res);
    } catch (error) {
      setResult({ error: 'Failed to analyze threat' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <ScrollView contentContainerStyle={styles.container}>
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Threat Analyzer</Text>
          <Text style={styles.headerSubtitle}>Analyze text for potential threats</Text>
        </View>

        <View style={styles.inputCard}>
          <Text style={styles.label}>Source</Text>
          <View style={styles.sourceSelector}>
            {['reddit', 'twitter', 'news'].map(s => (
              <TouchableOpacity
                key={s}
                style={[styles.sourceBtn, source === s && styles.sourceBtnActive]}
                onPress={() => setSource(s)}
              >
                <Text style={[styles.sourceBtnText, source === s && styles.sourceBtnTextActive]}>
                  {s.charAt(0).toUpperCase() + s.slice(1)}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          <Text style={styles.label}>Text Content</Text>
          <TextInput
            style={styles.input}
            multiline
            placeholder="Enter text to analyze..."
            placeholderTextColor="#64748B"
            value={text}
            onChangeText={setText}
          />
          
          <TouchableOpacity style={styles.analyzeBtn} onPress={handleAnalyze} disabled={loading || !text}>
            {loading ? <ActivityIndicator color="#FFF" /> : <Text style={styles.analyzeBtnText}>Analyze Threat</Text>}
          </TouchableOpacity>
        </View>

        {result && (
          <View style={styles.resultCard}>
            <Text style={styles.resultTitle}>Analysis Result</Text>
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
  sourceSelector: { flexDirection: 'row', marginBottom: 16 },
  sourceBtn: { paddingHorizontal: 12, paddingVertical: 8, borderRadius: 8, backgroundColor: '#334155', marginRight: 8 },
  sourceBtnActive: { backgroundColor: '#3B82F6' },
  sourceBtnText: { color: '#CBD5E1', fontSize: 14, fontWeight: '500' },
  sourceBtnTextActive: { color: '#FFF' },
  input: { backgroundColor: '#0F172A', borderRadius: 8, padding: 12, color: '#F8FAFC', minHeight: 120, textAlignVertical: 'top', borderWidth: 1, borderColor: '#334155', marginBottom: 16 },
  analyzeBtn: { backgroundColor: '#3B82F6', borderRadius: 8, padding: 16, alignItems: 'center' },
  analyzeBtnText: { color: '#FFF', fontSize: 16, fontWeight: 'bold' },
  resultCard: { backgroundColor: '#1E293B', borderRadius: 16, padding: 16, borderWidth: 1, borderColor: '#334155' },
  resultTitle: { fontSize: 16, fontWeight: 'bold', color: '#F8FAFC', marginBottom: 12 },
  resultText: { color: '#CBD5E1', fontFamily: 'monospace' },
});
