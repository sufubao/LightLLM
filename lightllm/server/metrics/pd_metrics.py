"""Collect registered P/D nodes without losing their individual metric series."""

import asyncio

import httpx
from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.parser import text_string_to_metric_families

from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
SCRAPE_TIMEOUT = 5.0
SCRAPE_CONCURRENCY = 16


async def collect_pd_metrics(nodes, roles):
    semaphore = asyncio.Semaphore(SCRAPE_CONCURRENCY)

    async def fetch(client, node):
        async with semaphore:
            response = await client.get(f"http://{node.client_ip_port}/metrics")
            response.raise_for_status()
            families = list(text_string_to_metric_families(response.text))
            if not any(family.samples for family in families):
                raise ValueError("Empty metrics response")
            seen = set()
            for family in families:
                if family.name.startswith("lightllm_pd_scrape_") or family.name == "lightllm_pd_registered_nodes":
                    raise ValueError("Reserved PD monitoring metric name")
                for sample in family.samples:
                    if "pd_role" in sample.labels or "pd_node" in sample.labels:
                        raise ValueError("Reserved PD monitoring label")
                    key = (sample.name, tuple(sorted(sample.labels.items())))
                    if key in seen:
                        raise ValueError(f"Duplicate metric sample: {sample.name}")
                    seen.add(key)
                family.samples = [
                    sample._replace(labels={**sample.labels, "pd_role": node.mode, "pd_node": node.client_ip_port})
                    for sample in family.samples
                ]
            return families

    async with httpx.AsyncClient(timeout=SCRAPE_TIMEOUT, trust_env=False, follow_redirects=False) as client:
        results = await asyncio.gather(
            *(asyncio.wait_for(fetch(client, node), timeout=SCRAPE_TIMEOUT) for node in nodes),
            return_exceptions=True,
        )

    merged = {}
    sample_types = {}
    success = GaugeMetricFamily(
        "lightllm_pd_scrape_success",
        "Whether a registered P/D node was scraped successfully",
        labels=["pd_role", "pd_node"],
    )
    for node, result in zip(nodes, results):
        if not isinstance(result, BaseException):
            # Validate the entire node before merging, so failed nodes contribute no partial data.
            types = {name: family.type for name, family in merged.items()}
            node_sample_types = dict(sample_types)
            for family in result:
                if (family.name in types and types[family.name] != family.type) or any(
                    sample.name in node_sample_types and node_sample_types[sample.name] != family.type
                    for sample in family.samples
                ):
                    result = ValueError(f"Conflicting metric type: {family.name}")
                    break
                types[family.name] = family.type
                node_sample_types.update((sample.name, family.type) for sample in family.samples)
        ok = not isinstance(result, BaseException)
        success.add_metric([node.mode, node.client_ip_port], int(ok))
        if not ok:
            logger.warning(f"P/D metrics scrape failed for {node.mode} {node.client_ip_port}: {result!r}")
            continue
        sample_types = node_sample_types
        for family in result:
            if family.name in merged:
                merged[family.name].samples.extend(family.samples)
            else:
                merged[family.name] = family

    registered = GaugeMetricFamily(
        "lightllm_pd_registered_nodes", "Number of registered P/D nodes in this scrape", labels=["pd_role"]
    )
    for role in roles:
        registered.add_metric([role], sum(node.mode == role for node in nodes))

    class Collector:
        def collect(self):
            yield from merged.values()
            yield success
            yield registered

    registry = CollectorRegistry()
    registry.register(Collector())
    return generate_latest(registry)
