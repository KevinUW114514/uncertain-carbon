from __future__ import annotations

import datetime as dt
import time
from typing import Dict, Optional, Tuple, List

from kubernetes import client, config
from kubernetes.client import ApiException

# NAMESPACE = "default"
# DEPLOYMENT = "newdeploy-ml-image-processing-default-a616-8254b24ce685"
# CONTAINER = "ml-image-processing"

# REQUESTS = {"cpu": "500m", "memory": "100Mi"}
# LIMITS   = {"cpu": "2",    "memory": "100Mi"}

# TIMEOUT_S = 600
# POLL_S = 2.0


def load_config() -> None:
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()


def get_container_resources_from_podspec(pod_spec: client.V1PodSpec, container_name: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    for c in pod_spec.containers or []:
        if c.name == container_name:
            req = (c.resources.requests or {}) if c.resources else {}
            lim = (c.resources.limits or {}) if c.resources else {}
            # Convert to plain dict[str,str]
            return dict(req), dict(lim)
    raise KeyError(f"Container '{container_name}' not found in pod spec")


import re
import time
from typing import Dict, Tuple, Optional

_CPU_RE = re.compile(r"^\s*(\d+)(m?)\s*$")
_MEM_RE = re.compile(r"^\s*(\d+)\s*([KMGTE]i|[kMGTPE]?|)\s*$")

# Binary (IEC) units
_IEC = {
    "Ki": 1024,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "Pi": 1024**5,
    "Ei": 1024**6,
}
# Decimal (SI) units often seen in K8s quantities
_SI = {
    "": 1,
    "k": 10**3,
    "M": 10**6,
    "G": 10**9,
    "T": 10**12,
    "P": 10**15,
    "E": 10**18,
}

def _parse_cpu_to_millicores(q: Optional[str]) -> Optional[int]:
    """Parse cpu quantity string into millicores (e.g., '2000m'->2000, '2'->2000)."""
    if q is None:
        return None
    s = str(q).strip()
    m = _CPU_RE.match(s)
    if not m:
        raise ValueError(f"Unrecognized cpu quantity: {q!r}")
    val = int(m.group(1))
    is_m = (m.group(2) == "m")
    return val if is_m else val * 1000

def _parse_mem_to_bytes(q: Optional[str]) -> Optional[int]:
    """Parse memory quantity string into bytes (e.g., '350Mi'->367001600)."""
    if q is None:
        return None
    s = str(q).strip()
    m = _MEM_RE.match(s)
    if not m:
        raise ValueError(f"Unrecognized memory quantity: {q!r}")
    val = int(m.group(1))
    unit = m.group(2)

    # IEC
    if unit in _IEC:
        return val * _IEC[unit]

    # SI (including empty == bytes)
    # Kubernetes also allows 'm' for milli-bytes in some contexts, but it's not used for memory requests/limits.
    if unit in _SI:
        return val * _SI[unit]

    raise ValueError(f"Unrecognized memory unit in quantity: {q!r}")

def _normalize_resource(key: str, q: Optional[str]) -> Optional[int]:
    """Return a canonical integer for comparison."""
    if q is None:
        return None
    if key == "cpu":
        return _parse_cpu_to_millicores(q)
    if key == "memory":
        return _parse_mem_to_bytes(q)
    # For other resources, fall back to raw string compare by hashing
    # (you can extend this if you add ephemeral-storage, hugepages, etc.)
    return None  # signal: use string compare

def verify_deployment_resources(
    apps,
    namespace: str,
    name: str,
    container_name: str,
    expected_requests: Dict[str, str],
    expected_limits: Dict[str, str],
    timeout_s: int = 10,
    poll_s: float = 1.0,
) -> None:
    """
    Verify the Deployment's pod template spec contains the expected resources.
    Normalizes cpu/memory quantities so semantically equivalent values compare equal
    (e.g., '2' == '2000m', '1024Mi' == '1Gi').
    """
    start = time.time()
    last_req: Dict[str, str] = {}
    last_lim: Dict[str, str] = {}

    while True:
        dep = apps.read_namespaced_deployment(name=name, namespace=namespace)
        req, lim = get_container_resources_from_podspec(dep.spec.template.spec, container_name)

        last_req, last_lim = req or {}, lim or {}

        mismatches = []

        def _check(expected: Dict[str, str], observed: Dict[str, str], label: str) -> None:
            for k, exp_v in expected.items():
                obs_v = observed.get(k)

                norm_exp = _normalize_resource(k, exp_v)
                norm_obs = _normalize_resource(k, obs_v)

                if norm_exp is None or norm_obs is None:
                    # Fallback to string compare for unknown resources
                    if obs_v != exp_v:
                        mismatches.append(f"{label}.{k}: expected {exp_v!r}, observed {obs_v!r}")
                else:
                    if norm_exp != norm_obs:
                        # Provide both canonical and original values for debugging
                        mismatches.append(
                            f"{label}.{k}: expected {exp_v!r} ({norm_exp}), observed {obs_v!r} ({norm_obs})"
                        )

        _check(expected_requests, last_req, "requests")
        _check(expected_limits, last_lim, "limits")

        if not mismatches:
            return

        if time.time() - start > timeout_s:
            raise RuntimeError(
                f"Deployment resource verification failed for {namespace}/{name} container={container_name}. "
                f"Mismatches: {', '.join(mismatches)}. "
                f"Expected requests={expected_requests}, limits={expected_limits}. "
                f"Observed requests={last_req}, limits={last_lim}"
            )

        time.sleep(poll_s)



def rollout_restart(apps: client.AppsV1Api, namespace: str, name: str) -> str:
    """
    Trigger restart by patching restartedAt annotation. Returns the timestamp used.
    """
    ts = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    body = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {"kubectl.kubernetes.io/restartedAt": ts}
                }
            }
        }
    }
    apps.patch_namespaced_deployment(name=name, namespace=namespace, body=body)
    return ts


def wait_for_deployment_rollout(apps: client.AppsV1Api, namespace: str, name: str,
                                timeout_s: int = 300, poll_s: float = 2.0) -> None:
    start = time.time()
    while True:
        dep = apps.read_namespaced_deployment(name=name, namespace=namespace)

        desired = dep.spec.replicas or 0
        status = dep.status

        observed_gen = status.observed_generation or 0
        current_gen = dep.metadata.generation or 0

        updated = status.updated_replicas or 0
        available = status.available_replicas or 0
        unavailable = status.unavailable_replicas or 0

        # Fail fast if ProgressDeadlineExceeded
        if status.conditions:
            for c in status.conditions:
                if c.type == "Progressing" and c.status == "False" and c.reason == "ProgressDeadlineExceeded":
                    raise RuntimeError(
                        f"Rollout failed: ProgressDeadlineExceeded for {namespace}/{name}. message={c.message}"
                    )

        done = (
            observed_gen >= current_gen
            and updated >= desired
            and available >= desired
            and unavailable == 0
        )

        print(
            f"[rollout] desired={desired} updated={updated} available={available} "
            f"unavailable={unavailable} gen={current_gen} observedGen={observed_gen}"
        )

        if done:
            return

        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for rollout of {namespace}/{name}. "
                f"desired={desired}, updated={updated}, available={available}, "
                f"unavailable={unavailable}, observedGen={observed_gen}, gen={current_gen}"
            )

        time.sleep(poll_s)


def verify_new_replicaset_template_resources(
    apps,
    core,
    namespace: str,
    deployment_name: str,
    container_name: str,
    expected_requests: Dict[str, str],
    expected_limits: Dict[str, str],
) -> None:
    """
    After rollout, verify that the *new ReplicaSet* owned by this Deployment has the expected resources
    in its pod template spec. Normalizes cpu/memory quantities so semantically equivalent values compare equal.
    """
    dep = apps.read_namespaced_deployment(name=deployment_name, namespace=namespace)

    selector = dep.spec.selector.match_labels or {}
    if not selector:
        raise RuntimeError("Deployment has no match_labels selector; cannot reliably find ReplicaSets")

    label_selector = ",".join([f"{k}={v}" for k, v in selector.items()])

    rs_list = apps.list_namespaced_replica_set(namespace=namespace, label_selector=label_selector)

    owned: List = []
    for rs in rs_list.items:
        for o in (rs.metadata.owner_references or []):
            if o.kind == "Deployment" and o.name == deployment_name:
                owned.append(rs)
                break

    if not owned:
        raise RuntimeError(f"No ReplicaSets found owned by Deployment {namespace}/{deployment_name}")

    # Prefer the RS that matches the Deployment's pod-template-hash (most reliable),
    # else fall back to newest by creationTimestamp.
    dep_hash = None
    try:
        dep_hash = (dep.spec.template.metadata.labels or {}).get("pod-template-hash")
    except Exception:
        dep_hash = None

    newest = None
    if dep_hash:
        for rs in owned:
            rs_hash = (rs.spec.template.metadata.labels or {}).get("pod-template-hash")
            if rs_hash == dep_hash:
                newest = rs
                break

    if newest is None:
        owned.sort(key=lambda r: r.metadata.creation_timestamp or dt.datetime.min, reverse=True)
        newest = owned[0]

    req, lim = get_container_resources_from_podspec(newest.spec.template.spec, container_name)
    req = req or {}
    lim = lim or {}

    mismatches = []

    def _check(expected: Dict[str, str], observed: Dict[str, str], label: str) -> None:
        for k, exp_v in expected.items():
            obs_v = observed.get(k)

            norm_exp = _normalize_resource(k, exp_v)
            norm_obs = _normalize_resource(k, obs_v)

            if norm_exp is None or norm_obs is None:
                if obs_v != exp_v:
                    mismatches.append(f"{label}.{k}: expected {exp_v!r}, observed {obs_v!r}")
            else:
                if norm_exp != norm_obs:
                    mismatches.append(
                        f"{label}.{k}: expected {exp_v!r} ({norm_exp}), observed {obs_v!r} ({norm_obs})"
                    )

    _check(expected_requests, req, "requests")
    _check(expected_limits, lim, "limits")

    if mismatches:
        raise RuntimeError(
            f"ReplicaSet template verification failed. RS={newest.metadata.name} "
            f"Mismatches: {', '.join(mismatches)}. "
            f"Expected requests={expected_requests}, limits={expected_limits}. "
            f"Observed requests={req}, limits={lim}"
        )

    # Optional: verify pods created by the newest RS exist and are Ready (best-effort)
    rs_selector = newest.spec.selector.match_labels or {}
    if rs_selector:
        pod_selector = ",".join([f"{k}={v}" for k, v in rs_selector.items()])
        pods = core.list_namespaced_pod(namespace=namespace, label_selector=pod_selector).items
        ready = 0
        for p in pods:
            conds = (p.status.conditions or []) if p.status else []
            if any(c.type == "Ready" and c.status == "True" for c in conds):
                ready += 1
        print(f"[verify] newest RS={newest.metadata.name} pods={len(pods)} ready={ready}")


# def update_kube_deployment(
#     container_name, 
#     requests, 
#     limits, 
#     deployment_name, 
#     namespace="default", 
#     timeout_s=60, 
#     poll_s=0.5
# ) -> None:
#     load_config()

#     apps = client.AppsV1Api()
#     core = client.CoreV1Api()

#     # 1) Patch resources
#     patch_body = {
#         "spec": {
#             "template": {
#                 "spec": {
#                     "containers": [{
#                         "name": container_name,
#                         "resources": {"requests": requests, "limits": limits},
#                     }]
#                 }
#             }
#         }
#     }
#     apps.patch_namespaced_deployment(name=deployment_name, namespace=namespace, body=patch_body)
#     print("[ok] Patched Deployment resources")

#     # 2) Verify Deployment spec reflects the change
#     verify_deployment_resources(apps, namespace, deployment_name, container_name, requests, limits)
#     dep = apps.read_namespaced_deployment(name=deployment_name, namespace=namespace)
#     req, lim = get_container_resources_from_podspec(dep.spec.template.spec, container_name)
#     print(f"[ok] Verified Deployment template resources: requests={req} limits={lim}")

#     # 3) Rollout restart
#     ts = rollout_restart(apps, namespace, deployment_name)
#     print(f"[ok] Triggered rollout restart restartedAt={ts}")

#     # 4) Wait for rollout completion
#     wait_for_deployment_rollout(apps, namespace, deployment_name, timeout_s=timeout_s, poll_s=poll_s)
#     print("[ok] Rollout completed")

#     # 5) Verify newest ReplicaSet template resources (stronger evidence pods will have it)
#     verify_new_replicaset_template_resources(apps, core, namespace, deployment_name, container_name, requests, limits)
#     print("[ok] Verified newest ReplicaSet template resources match expected")

def delete_hpa_for_deployment(autoscaling, namespace: str, deployment_name: str):
    # You need to know the HPA name. For NewDeploy it’s usually derived from the function/deployment.
    # If you don’t know it, list and filter HPAs by scaleTargetRef.
    hpas = autoscaling.list_namespaced_horizontal_pod_autoscaler(namespace=namespace).items
    for hpa in hpas:
        ref = hpa.spec.scale_target_ref
        if ref and ref.kind == "Deployment" and ref.name == deployment_name:
            try:
                autoscaling.delete_namespaced_horizontal_pod_autoscaler(
                    name=hpa.metadata.name, namespace=namespace
                )
                print(f"[ok] Deleted HPA {hpa.metadata.name}")
            except ApiException as e:
                print(f"[warn] Failed to delete HPA {hpa.metadata.name}: {e}")


def update_kube_deployment(
    container_name,
    requests,
    limits,
    deployment_name,
    namespace="default",
    timeout_s=60,
    poll_s=0.5,
) -> None:
    load_config()

    apps = client.AppsV1Api()
    core = client.CoreV1Api()

    # Helper: build label selector from a Deployment's selector.matchLabels
    def _deployment_label_selector(dep_obj) -> str:
        ml = (dep_obj.spec.selector.match_labels or {})
        if not ml:
            raise ValueError(
                f"Deployment {namespace}/{deployment_name} has empty selector.matchLabels; "
                "cannot safely target pods for scale-to-zero verification."
            )
        return ",".join([f"{k}={v}" for k, v in ml.items()])

    # Helper: wait until zero pods remain for a selector
    def _wait_for_zero_pods(label_selector: str) -> None:
        deadline = time.monotonic() + timeout_s
        last_n = None
        while time.monotonic() < deadline:
            pods = core.list_namespaced_pod(
                namespace=namespace, label_selector=label_selector
            ).items
            n = len(pods)
            last_n = n
            if n == 0:
                return
            time.sleep(poll_s)
        raise TimeoutError(
            f"Timed out waiting for pods to reach 0 for selector '{label_selector}' "
            f"in {namespace}; last_count={last_n}"
        )

    # Helper: scale deployment replicas
    def _scale_replicas(replicas: int) -> None:
        body = {"spec": {"replicas": replicas}}
        apps.patch_namespaced_deployment_scale(
            name=deployment_name, namespace=namespace, body=body
        )

    # Read deployment once to capture current replicas + selector for verification
    dep_before = apps.read_namespaced_deployment(name=deployment_name, namespace=namespace)
    # original_replicas = int(dep_before.spec.replicas or 0)
    if container_name == "ml-image-processing":
        original_replicas = 17
    else:
        original_replicas = 49
        
    label_selector = _deployment_label_selector(dep_before)
    print(f"[info] Original replicas={original_replicas}, selector='{label_selector}'")

    # 1) Patch resources (updates Pod template => new ReplicaSet template)
    patch_body = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "name": container_name,
                            "resources": {"requests": requests, "limits": limits},
                        }
                    ]
                }
            }
        }
    }
    apps.patch_namespaced_deployment(
        name=deployment_name, namespace=namespace, body=patch_body
    )
    print("[ok] Patched Deployment resources")

    # 2) Verify Deployment spec reflects the change
    verify_deployment_resources(
        apps, namespace, deployment_name, container_name, requests, limits
    )
    dep = apps.read_namespaced_deployment(name=deployment_name, namespace=namespace)
    req, lim = get_container_resources_from_podspec(dep.spec.template.spec, container_name)
    print(f"[ok] Verified Deployment template resources: requests={req} limits={lim}")

    # === Option A: scale to 0 then back up (recreate ALL pods with new template) ===
    # 3) Scale down to 0 (terminates all pods)
    print("[info] Scaling deployment to 0 replicas to force full pod replacement")
    _scale_replicas(0)
    _wait_for_zero_pods(label_selector)
    print("[ok] All pods terminated (replicas=0)")

    # 4) Scale back to original replicas (creates all pods using the patched template)
    print(f"[info] Scaling deployment back to {original_replicas} replicas")
    _scale_replicas(original_replicas)

    # 5) Wait for rollout completion (ensures new pods are ready)
    wait_for_deployment_rollout(
        apps, namespace, deployment_name, timeout_s=timeout_s, poll_s=poll_s
    )
    print("[ok] Rollout completed after scale-to-zero")

    delete_hpa_for_deployment(
        autoscaling=client.AutoscalingV1Api(),
        namespace=namespace,
        deployment_name=deployment_name,
    )
    
    # 6) Verify newest ReplicaSet template resources (strong evidence pods will have it)
    verify_new_replicaset_template_resources(
        apps, core, namespace, deployment_name, container_name, requests, limits
    )
    print("[ok] Verified newest ReplicaSet template resources match expected")

def reset(): 
    NAMESPACE = "default"
    DEPLOYMENT = "newdeploy-ml-image-processing-default-a616-8254b24ce685"
    CONTAINER = "ml-image-processing"

    REQUESTS = {"cpu": "500m", "memory": "60Mi"}
    LIMITS   = {"cpu": "800m",    "memory": "60Mi"}

    TIMEOUT_S = 600
    POLL_S = 2.0
    
    update_kube_deployment(
        namespace=NAMESPACE,
        deployment_name=DEPLOYMENT,
        container_name=CONTAINER,
        requests=REQUESTS,
        limits=LIMITS,
        timeout_s=TIMEOUT_S,
        poll_s=POLL_S
    )

    NAMESPACE = "default"
    DEPLOYMENT = "newdeploy-ml-object-detection-default-beef-7bcaa92aa890"
    CONTAINER = "yolo"

    REQUESTS = {"cpu": "2000m", "memory": "400Mi"}
    LIMITS   = {"cpu": "6000m",    "memory": "400Mi"}

    TIMEOUT_S = 600
    POLL_S = 2.0
    
    update_kube_deployment(
        namespace=NAMESPACE,
        deployment_name=DEPLOYMENT,
        container_name=CONTAINER,
        requests=REQUESTS,
        limits=LIMITS,
        timeout_s=TIMEOUT_S,
        poll_s=POLL_S
    )

def test():
    NAMESPACE = "default"
    DEPLOYMENT = "newdeploy-ml-image-processing-default-a616-8254b24ce685"
    CONTAINER = "ml-image-processing"

    # REQUESTS = {"cpu": "300m", "memory": "50Mi"}
    # LIMITS   = {"cpu": "400m",    "memory": "54Mi"}

    REQUESTS = {"cpu": "300m", "memory": "50Mi"}
    LIMITS   = {"cpu": "426m",    "memory": "66Mi"}

    TIMEOUT_S = 600
    POLL_S = 2.0
    
    update_kube_deployment(
        namespace=NAMESPACE,
        deployment_name=DEPLOYMENT,
        container_name=CONTAINER,
        requests=REQUESTS,
        limits=LIMITS,
        timeout_s=TIMEOUT_S,
        poll_s=POLL_S
    )

    NAMESPACE = "default"
    DEPLOYMENT = "newdeploy-ml-object-detection-default-beef-7bcaa92aa890"
    CONTAINER = "yolo"

    # REQUESTS = {"cpu": "1500m", "memory": "400Mi"}
    # LIMITS   = {"cpu": "3800m",    "memory": "467Mi"}

    REQUESTS = {"cpu": "1500m", "memory": "400Mi"}
    LIMITS   = {"cpu": "4286m",    "memory": "447Mi"}

    TIMEOUT_S = 600
    POLL_S = 2.0
    
    update_kube_deployment(
        namespace=NAMESPACE,
        deployment_name=DEPLOYMENT,
        container_name=CONTAINER,
        requests=REQUESTS,
        limits=LIMITS,
        timeout_s=TIMEOUT_S,
        poll_s=POLL_S
    )

if __name__ == "__main__":
    # reset()
    test()