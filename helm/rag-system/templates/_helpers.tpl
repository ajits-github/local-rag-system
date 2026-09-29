{{/*
Chart name, truncated/DNS-safe (standard `helm create` scaffold pattern).
*/}}
{{- define "rag-system.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Full release-qualified name, e.g. "myrelease-rag-system". Used as a prefix
for any resource name that isn't a fixed, well-known Service name (rag-api,
frontend, postgres, ollama-host keep their fixed names -- see the top of
templates/rag-api/service.yaml for why).
*/}}
{{- define "rag-system.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Common labels applied to every object's metadata.labels. Deliberately NOT
used for spec.selector/matchLabels -- those must stay stable for the life of
a Deployment/StatefulSet (Kubernetes rejects changing them on an existing
object), so selector labels are the fixed "app: <component>" values below,
set directly in each workload template instead of through this helper.
*/}}
{{- define "rag-system.labels" -}}
helm.sh/chart: {{ include "rag-system.chart" . }}
{{ include "rag-system.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "rag-system.selectorLabels" -}}
app.kubernetes.io/name: {{ include "rag-system.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "rag-system.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
ServiceAccount name: the release-qualified one when we create it, "default"
when serviceAccount.create is false and no override was given.
*/}}
{{- define "rag-system.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "rag-system.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Name of the Secret the workloads read DATABASE_URL/JWT_HS256_SECRET/
COHERE_API_KEY from -- either the chart-rendered one, or an operator-
supplied existingSecret (see values-prod.yaml and README's "Secret
hardening" section).
*/}}
{{- define "rag-system.secretName" -}}
{{- if .Values.secret.existingSecret }}
{{- .Values.secret.existingSecret }}
{{- else }}
{{- printf "%s-secret" (include "rag-system.fullname" .) }}
{{- end }}
{{- end }}
