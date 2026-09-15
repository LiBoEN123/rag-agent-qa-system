import type { CandidateDetailResponse, IngestionCandidateStatus, IngestionIndexRun } from '../api/types'

export function candidateDetail(status: IngestionCandidateStatus = 'pending'): CandidateDetailResponse {
  return {
    summary: {
      candidate_id: 'cand_example', title: '测试文档', source_type: 'github',
      source_uri: 'https://github.com/example/docs/blob/main/guide.md',
      content_hash: 'sha256:example', content_length: 3000, trust_level: 'untrusted_external',
      policy_decision: 'require_approval', policy_reasons: ['untrusted_external'],
      status, approval_status: status === 'approved' ? 'approved' : 'pending', requested_at: '2026-09-11T10:00:00Z',
    },
    requester: { principal_id: 'agent-user', principal_type: 'user', source: 'agent-tool-promotion' },
    approval: { status: status === 'approved' ? 'approved' : 'pending', decided_at: null, decided_by: null, reason: null },
    safe_metadata: { path: 'guide.md' }, content_preview: '<script>untrusted()</script>安全预览',
    content_length: 3000, preview_truncated: true, content_is_untrusted: true, latest_index_result: null,
  }
}

export function indexRun(): IngestionIndexRun {
  return {
    id: 7, candidate_id: 'cand_example', operation: 'index', status: 'success',
    started_at: '2026-09-11T10:10:00Z', completed_at: '2026-09-11T10:10:01Z',
    added: 2, updated: 3, deleted: 4, skipped: 5, error_code: null, error_message: null,
  }
}
