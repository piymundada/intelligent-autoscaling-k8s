package com.example.springworkloadsimulator.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Generates controllable CPU and heap pressure so load tests produce
 * realistic JVM signals (GC pauses, heap growth, thread activity).
 */
@RestController
@RequestMapping("/api/load")
public class LoadController {

    /**
     * CPU-bound work: repeated SHA-256 hashing.
     * @param iterations number of hash rounds (default 500)
     */
    @GetMapping("/cpu")
    public Map<String, Object> cpuLoad(@RequestParam(defaultValue = "500") int iterations)
            throws NoSuchAlgorithmException {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        String input = "jvm-autoscaling-thesis";
        for (int i = 0; i < iterations; i++) {
            input = bytesToHex(digest.digest(input.getBytes()));
        }
        Map<String, Object> result = new HashMap<>();
        result.put("iterations", iterations);
        result.put("hash", input.substring(0, 16));
        return result;
    }

    /**
     * Heap-allocation work: allocates short-lived objects to trigger GC.
     * @param objects number of 1 KB string objects to allocate (default 5000)
     */
    @GetMapping("/memory")
    public Map<String, Object> memoryLoad(@RequestParam(defaultValue = "5000") int objects) {
        List<String> sink = new ArrayList<>(objects);
        for (int i = 0; i < objects; i++) {
            sink.add("x".repeat(1024) + i);
        }
        long sum = sink.stream().mapToLong(String::length).sum();
        Map<String, Object> result = new HashMap<>();
        result.put("objectsAllocated", objects);
        result.put("bytesAllocated", sum);
        return result;
    }

    /**
     * GC-pressure work: allocates large short-lived byte chunks so Young GC
     * runs frequently while CPU stays low: latency degrades from GC pauses,
     * not CPU saturation (the signal infra-only autoscalers cannot see).
     * @param cycles  number of allocation cycles (default 20)
     * @param chunkKb size of each chunk in KB (default 4096 = 4 MB)
     */
    @GetMapping("/gc")
    public Map<String, Object> gcLoad(
            @RequestParam(defaultValue = "20") int cycles,
            @RequestParam(defaultValue = "4096") int chunkKb) {
        long start = System.currentTimeMillis();
        long totalBytes = 0;
        for (int i = 0; i < cycles; i++) {
            byte[] chunk = new byte[chunkKb * 1024];
            // Touch first/last bytes so allocation is not optimised away;
            // the chunk goes out of scope each cycle -> short-lived garbage.
            chunk[0] = 1;
            chunk[chunk.length - 1] = 1;
            totalBytes += chunk.length;
        }
        Map<String, Object> result = new HashMap<>();
        result.put("cycles", cycles);
        result.put("chunkKb", chunkKb);
        result.put("totalAllocatedMb", totalBytes / (1024 * 1024));
        result.put("elapsedMs", System.currentTimeMillis() - start);
        return result;
    }

    /**
     * Mixed load: combined CPU + heap pressure.
     */
    @GetMapping("/mixed")
    public Map<String, Object> mixedLoad(
            @RequestParam(defaultValue = "300") int iterations,
            @RequestParam(defaultValue = "2000") int objects) throws NoSuchAlgorithmException {
        Map<String, Object> cpu = cpuLoad(iterations);
        Map<String, Object> mem = memoryLoad(objects);
        Map<String, Object> result = new HashMap<>();
        result.put("cpu", cpu);
        result.put("memory", mem);
        return result;
    }

    private static String bytesToHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }
}
